import os
import json
import httpx

API_BASE = os.getenv("EVAL_API_BASE", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")  # reuse the same env var you use for the server

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
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    results = []

    with httpx.Client(timeout=60.0) as client:
        for t in TESTS:
            payload = {"question": t["question"], "top_k": 3}
            r = client.post(f"{API_BASE}/query", headers=headers, json=payload)

            record = {
                "test": t["name"],
                "question": t["question"],
                "expected_abstained": t["expect_abstain"],
                "http_status": r.status_code,
            }

            if r.status_code != 200:
                record["error"] = r.text
                results.append(record)
                continue

            data = r.json()
            grounding = data.get("grounding", {})
            citations = data.get("citations", [])
            answer = data.get("answer", "")
            
            has_citation = any(cid in answer for cid in citations)
            
            record.update(
                {
                    "abstained": grounding.get("abstained"),
                    "best_distance": grounding.get("best_distance"),
                    "threshold": grounding.get("max_distance_threshold"),
                    "warning_flags": data.get("warning_flags", []),
                    "has_citation_in_answer": has_citation if not grounding.get("abstained") else None,
                }
            )
            results.append(record)

    total = len(results)
    ok = sum(1 for r in results if r.get("http_status") == 200)
    correct_abstain = sum(
        1 for r in results
        if r.get("http_status") == 200
        and r.get("abstained") == r.get("expected_abstained")
    )

    print(json.dumps(results, indent=2))
    print("\nSUMMARY")
    print(
        json.dumps(
            {
                "total": total,
                "http_200": ok,
                "abstain_matches_expectation": correct_abstain,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


