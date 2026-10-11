"""Strict JSON experiment allowlist. Proposals are data, never executable code."""

import json
import math

from src.features import FEATURE_NAMES

ROLES = ("features", "classifier", "robustness")
FEATURE_SETS = ("all", "mfcc", "spectral", "no_energy")
ROLE_INSTRUCTIONS = {
    "features": "Compare MFCC, spectral, or no_energy subsets. Keep logistic regression C=1 and class_weight=null.",
    "classifier": "Compare bounded logistic regression or random forest settings using all features.",
    "robustness": "Reduce level/recording shortcuts: use no_energy features, balanced logistic regression, and bounded C. Vendor diagnostics describe seen-vendor validation only.",
}


class ProposalError(ValueError):
    """An experiment violates the fixed allowlist."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProposalError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def parse_json(text: str):
    if not isinstance(text, str) or len(text) > 16384:
        raise ProposalError("Expected a JSON object of at most 16 KiB.")
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProposalError("Non-finite JSON is not allowed.")))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ProposalError("Response must be plain JSON, without code fences or extra text.") from exc


def validate_proposal(value, role: str) -> dict:
    if role not in ROLES or not isinstance(value, dict) or set(value) != {"role", "feature_set", "classifier", "parameters"}:
        raise ProposalError("Expected exactly role, feature_set, classifier, and parameters.")
    if value["role"] != role or value["feature_set"] not in FEATURE_SETS:
        raise ProposalError("Role mismatch or unknown feature set.")
    params = value["parameters"]
    if not isinstance(params, dict):
        raise ProposalError("Parameters must be an object.")
    if value["classifier"] == "logistic_regression":
        if set(params) != {"C", "class_weight"}:
            raise ProposalError("Logistic parameters must be C and class_weight.")
        c = params["C"]
        if type(c) not in (float, int) or not 0.01 <= c <= 10 or not math.isfinite(c):
            raise ProposalError("Logistic C must be finite and between 0.01 and 10.")
    elif value["classifier"] == "random_forest":
        if set(params) != {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"}:
            raise ProposalError("Random forest parameters do not match the allowlist.")
        for name, allowed in (("n_estimators", (50, 100)), ("max_depth", (3, 6, 10)), ("min_samples_leaf", (1, 3, 5))):
            if type(params[name]) is not int or params[name] not in allowed:
                raise ProposalError("Random forest size/depth/leaf settings are outside the allowlist.")
        if params["max_features"] not in ("sqrt", "log2"):
            raise ProposalError("Random forest max_features must be sqrt or log2.")
    else:
        raise ProposalError("Unknown classifier; only logistic_regression and random_forest are allowed.")
    if params["class_weight"] not in (None, "balanced"):
        raise ProposalError("class_weight must be null or balanced.")
    if role == "features" and (value["classifier"] != "logistic_regression"
                               or params["C"] != 1 or params["class_weight"] is not None):
        raise ProposalError("Features role must keep the baseline classifier settings.")
    if role == "classifier" and value["feature_set"] != "all":
        raise ProposalError("Classifier role must use all features.")
    if role == "robustness" and (value["classifier"] != "logistic_regression"
                                 or value["feature_set"] != "no_energy" or params["class_weight"] != "balanced"):
        raise ProposalError("Robustness role requires no_energy features and balanced logistic regression.")
    return json.loads(json.dumps(value, allow_nan=False))


def feature_indices(name: str) -> list[int]:
    if name not in FEATURE_SETS:
        raise ProposalError("Unknown feature set.")
    return [i for i, feature in enumerate(FEATURE_NAMES)
            if name == "all" or (name == "mfcc" and feature.startswith("mfcc_"))
            or (name == "spectral" and not feature.startswith("mfcc_"))
            or (name == "no_energy" and not feature.startswith(("rms_", "mfcc_00_")))]


def static_proposal(role: str, round_number: int) -> dict:
    """Clearly marked offline fixtures; fits/metrics are still real."""
    index = round_number - 1
    proposal = {"role": role, "feature_set": "all", "classifier": "logistic_regression",
                "parameters": {"C": 1.0, "class_weight": None}}
    if role == "features":
        proposal["feature_set"] = ("mfcc", "spectral", "no_energy")[index]
    elif role == "classifier":
        if index == 1:
            proposal.update(classifier="random_forest", parameters={"n_estimators": 100, "max_depth": 6,
                            "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": None})
        else:
            proposal["parameters"]["C"] = (0.1, 1.0, 10.0)[index]
    elif role == "robustness":
        proposal.update(feature_set="no_energy", parameters={"C": (0.1, 1.0, 10.0)[index], "class_weight": "balanced"})
    return validate_proposal(proposal, role)


PROPOSAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
    "required": ["role", "feature_set", "classifier", "parameters"],
    "properties": {"role": {"enum": list(ROLES)}, "feature_set": {"enum": list(FEATURE_SETS)},
                   "classifier": {"enum": ["logistic_regression", "random_forest"]}, "parameters": {"type": "object"}},
    "allOf": [
        {"if": {"properties": {"classifier": {"const": "logistic_regression"}}},
         "then": {"properties": {"parameters": {"type": "object", "additionalProperties": False,
             "required": ["C", "class_weight"], "properties": {"C": {"type": "number", "minimum": 0.01, "maximum": 10},
             "class_weight": {"enum": [None, "balanced"]}}}}},
         "else": {"properties": {"parameters": {"type": "object", "additionalProperties": False,
             "required": ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"],
             "properties": {"n_estimators": {"type": "integer", "enum": [50, 100]}, "max_depth": {"type": "integer", "enum": [3, 6, 10]},
             "min_samples_leaf": {"type": "integer", "enum": [1, 3, 5]}, "max_features": {"enum": ["sqrt", "log2"]},
             "class_weight": {"enum": [None, "balanced"]}}}}}},
        {"if": {"properties": {"role": {"const": "features"}}}, "then": {"properties": {
            "classifier": {"const": "logistic_regression"}, "parameters": {"properties": {"C": {"const": 1}, "class_weight": {"const": None}}}}}},
        {"if": {"properties": {"role": {"const": "classifier"}}}, "then": {"properties": {"feature_set": {"const": "all"}}}},
        {"if": {"properties": {"role": {"const": "robustness"}}}, "then": {"properties": {
            "classifier": {"const": "logistic_regression"}, "feature_set": {"const": "no_energy"},
            "parameters": {"properties": {"class_weight": {"const": "balanced"}}}}}},
    ],
}
