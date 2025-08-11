import os, re, json, argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

import pandas as pd

# Optional LLM (LangChain + OpenAI). If no key, we stay rule-based.
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))

# --------- Dates ----------
def to_iso(d: datetime) -> str:
    return d.date().isoformat()

def start_of_week(d: datetime) -> datetime:
    # Monday start, UTC
    d = d.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return d - timedelta(days=(d.weekday() % 7))

def add_days(d: datetime, n: int) -> datetime:
    return d + timedelta(days=n)

def keyword_range(text: str, today: Optional[datetime]=None) -> Optional[Tuple[str,str]]:
    t = today or datetime.now(timezone.utc)
    sow = start_of_week(t)
    if re.search(r"last week", text, re.I):
        return to_iso(add_days(sow, -7)), to_iso(add_days(sow, -1))
    if re.search(r"(this|current) week", text, re.I):
        return to_iso(sow), to_iso(add_days(sow, 6))
    if re.search(r"next week", text, re.I):
        return to_iso(add_days(sow, 7)), to_iso(add_days(sow, 13))
    if re.search(r"\btoday\b", text, re.I):
        return to_iso(t), to_iso(t)
    if re.search(r"\byesterday\b", text, re.I):
        y = add_days(t, -1)
        return to_iso(y), to_iso(y)
    return None

def within(date_iso: str, rng: Optional[Tuple[str,str]]) -> bool:
    if not rng or not date_iso:
        return rng is None  # if no filter, accept all
    a, b = rng
    return a <= date_iso <= b

# --------- NLQ parsing ----------
def normalize(s: str) -> str:
    return (
        s
        .replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201C", '"').replace("\u201D", '"')
        .strip()
    )

def parse_query_rule_based(q: str) -> Dict:
    t = normalize(q).lower()
    intent = None

    has_hw = re.search(r"(homework|assignment|hw|work)", t) is not None
    submit_word = re.search(r"(submit|submitted|turn\s*in|turned\s*in|complete|done)", t) is not None
    pending_word = re.search(r"(pending|overdue|missing)", t) is not None
    neg = re.search(r"(haven't|hasn't|have not|has not|didn't|did not|not|no)", t) is not None

    if has_hw and (pending_word or re.search(r"not\s+submitted", t) or (neg and submit_word)):
        intent = "pending_homework"
    elif has_hw and submit_word and not neg:
        intent = "submitted_homework"
    elif re.search(r"\b(performance|avg(\s|-)?score|average|scores?)\b", t):
        intent = "performance"
    elif re.search(r"(upcoming.*quiz|quiz.*next|scheduled.*quiz|next week.*quiz)", t):
        intent = "upcoming_quizzes"

    # explicit range
    rng = keyword_range(t)
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(to|-|–|—)\s*(\d{4}-\d{2}-\d{2})", t)
    if m:
        rng = (m.group(1), m.group(3))

    return {"intent": intent, "date_range": rng}

# Optional LLM via LangChain (only if key present)
# Optional LLM via LangChain (only if key present)
def parse_query_llm(q: str) -> Dict:
    """LLM → strict JSON (no zod). Falls back to rule-based on any error."""
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.schema import StrOutputParser
    import json, re

    system = (
        "You convert short admin questions into JSON.\n"
        "Allowed intents: pending_homework, submitted_homework, performance, upcoming_quizzes.\n"
        "If the user says last/this/next week, output explicit Monday–Sunday ISO dates.\n"
        "Return ONLY compact JSON with keys: "
        '{"intent": "<one-of-4>", "dateRange": null or ["YYYY-MM-DD","YYYY-MM-DD"]}'
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", "Q: {q}\nReturn only JSON. No prose.")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"q": q}).strip()

    # Be defensive: extract first JSON object/array if any wrapper text sneaks in
    m = re.search(r'(\{.*\}|\[.*\])', raw, flags=re.S)
    txt = m.group(1) if m else raw

    try:
        data = json.loads(txt)
        rng = tuple(data["dateRange"]) if data.get("dateRange") else None
        return {"intent": data.get("intent"), "date_range": rng}
    except Exception:
        return parse_query_rule_based(q)

   
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.schema import StrOutputParser
    
    schema = z.object({
        "intent": z.enum(["pending_homework","submitted_homework","performance","upcoming_quizzes"]),
        "dateRange": z.union([z.tuple([z.string(), z.string()]), z.null()]),
    })
    parser = StructuredOutputParser.from_zod_schema(schema)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You map short admin questions to JSON. Use one of the intents. If user says last/this/next week, convert to explicit Mon–Sun ISO range based on current week."),
        ("user", "Q: {q}\nReturn only JSON:\n" + parser.get_format_instructions())
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | StrOutputParser()
    out = chain.invoke({"q": q})
    try:
        data = json.loads(out)
    except Exception:
        # fallback to rule-based if bad JSON
        return parse_query_rule_based(q)
    # Normalize field name
    return {"intent": data.get("intent"), "date_range": tuple(data["dateRange"]) if data.get("dateRange") else None}

def parse_query(q: str) -> Dict:
    if USE_LLM:
        try:
            return parse_query_llm(q)
        except Exception:
            return parse_query_rule_based(q)
    return parse_query_rule_based(q)

# --------- Data layer with RBAC ----------
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    # normalize types
    for col in ["grade"]:
        df[col] = df[col].astype(int)
    return df

def apply_scope(df: pd.DataFrame, scope: Dict) -> pd.DataFrame:
    f = df.copy()
    if scope.get("grade") is not None:
        f = f[f["grade"] == int(scope["grade"])]
    if scope.get("class"):
        f = f[f["class"] == str(scope["class"])]
    if scope.get("region"):
        f = f[f["region"] == str(scope["region"])]
    return f

def list_pending(df: pd.DataFrame, date_range: Optional[Tuple[str,str]]):
    f = df[df["submission_status"] == "pending"]
    if date_range:
        f = f[(f["homework_due_date"] >= date_range[0]) & (f["homework_due_date"] <= date_range[1])]
    cols = ["student_id","student_name","grade","class","region","homework_due_date","submission_status"]
    return f[cols].to_dict(orient="records")

def list_submitted(df: pd.DataFrame, date_range: Optional[Tuple[str,str]]):
    f = df[df["submission_status"] == "submitted"]
    if date_range:
        f = f[(f["submission_date"] >= date_range[0]) & (f["submission_date"] <= date_range[1])]
    cols = ["student_id","student_name","grade","class","region","submission_date","submission_status"]
    return f[cols].to_dict(orient="records")

def performance_summary(df: pd.DataFrame, date_range: Optional[Tuple[str,str]]):
    # explode quizzes
    rows = []
    for _, r in df.iterrows():
        dates = [d for d in r["quiz_dates"].split("|") if d]
        scores = [int(s) for s in r["quiz_scores"].split("|") if s]
        for d, s in zip(dates, scores):
            if (date_range is None) or (date_range[0] <= d <= date_range[1]):
                rows.append({"student_id": r["student_id"], "class": r["class"], "grade": int(r["grade"]), "score": s})
    if not rows:
        return {"students": 0, "quizzes": 0, "avg_score": None, "min_score": None, "max_score": None}, []
    import statistics as st
    scores = [r["score"] for r in rows]
    summary = {
        "students": len(set(r["student_id"] for r in rows)),
        "quizzes": len(rows),
        "avg_score": round(st.mean(scores), 1),
        "min_score": min(scores),
        "max_score": max(scores)
    }
    # breakdown by class
    breakdown = {}
    for r in rows:
        k = (r["grade"], r["class"])
        breakdown.setdefault(k, []).append(r["score"])
    breakdown_rows = [
        {"grade": g, "class": c, "avg_score": round(sum(v)/len(v), 1), "quizzes": len(v)}
        for (g, c), v in breakdown.items()
    ]
    return summary, breakdown_rows

def upcoming_quizzes(df: pd.DataFrame, date_range: Optional[Tuple[str,str]]):
    # Count students per upcoming quiz date
    counts = {}
    for _, r in df.iterrows():
        for d in [d for d in r["upcoming_quiz_dates"].split("|") if d]:
            if (date_range is None) or (date_range[0] <= d <= date_range[1]):
                counts[d] = counts.get(d, 0) + 1
    rows = [{"quiz_scheduled_date": d, "students": n} for d, n in sorted(counts.items())]
    return rows

# --------- Public API (callable + CLI) ----------
def answer(csv_path: str, scope: Dict, question: str) -> Dict:
    df = load_df(csv_path)
    df = apply_scope(df, scope)
    parsed = parse_query(question)
    intent, rng = parsed["intent"], parsed["date_range"]
    if not intent:
        return {"error": "could not parse intent", "debug": {"question": question}}

    if intent == "pending_homework":
        rows = list_pending(df, rng)
        return {"intent": intent, "filters": {"date_range": rng}, "count": len(rows), "rows": rows}
    if intent == "submitted_homework":
        rows = list_submitted(df, rng)
        return {"intent": intent, "filters": {"date_range": rng}, "count": len(rows), "rows": rows}
    if intent == "performance":
        summary, breakdown = performance_summary(df, rng)
        return {"intent": intent, "filters": {"date_range": rng}, "summary": summary, "breakdown": breakdown}
    if intent == "upcoming_quizzes":
        rows = upcoming_quizzes(df, rng)
        return {"intent": intent, "filters": {"date_range": rng}, "rows": rows, "count": len(rows)}
    return {"error": "unsupported intent", "intent": intent}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/students.csv")
    p.add_argument("--grade", type=int, required=True)
    p.add_argument("--class_", dest="class_", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("question", help="natural language question")
    args = p.parse_args()

    scope = {"grade": args.grade, "class": args.class_, "region": args.region}
    print(json.dumps(answer(args.data, scope, args.question), indent=2))
