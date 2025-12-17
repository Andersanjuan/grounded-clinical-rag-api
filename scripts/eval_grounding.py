import json
from app.rag.grounded_qa import answer_question


TESTS = [
    {
        "name": "in_scope_hand_hygiene",
        "question": "What are the key recommendations for hand hygiene?",
        "expect_abstain": False,
    },
    {
        "name": "in_scope_pressure_ulcers",
        "question": "How often should immobile patients be repositioned?",
        "expect_abstain": False,
    },
    {
        "name": "out_of_scope_antibiotic",
        "question": "What antibiotic should be prescribed for pneumonia?",
        "expect_abstain": True,
    },
]


def main():
    results = []
    for t in TESTS:
        r = answer_question(t["question"], top_k=3)
        grounding = r.get("grounding", {})
        abstained = grounding.get("abstained", None)

        results.append(
            {
                "test": t["name"],
                "question": t["question"],
                "expected_abstained": t["expect_abstain"],
                "abstained": abstained,
                "best_distance": grounding.get("best_distance"),
                "threshold": grounding.get("max_distance_threshold"),
                "warning_flags": r.get("warning_flags", []),
                "has_citation_in_answer": any(cid in r.get("answer", "") for cid in r.get("citations", [])),
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
