import os
import streamlit as st
from app import answer

st.set_page_config(page_title="Dumroo NLQ + RBAC", layout="centered")

st.title("NLQ + RBAC (Python)")
st.caption("Type a question in plain English. Scope limits your view.")

col1, col2, col3 = st.columns(3)
grade = col1.number_input("Grade", min_value=1, max_value=12, value=8, step=1)
clss  = col2.text_input("Class", value="A")
region= col3.selectbox("Region", ["East","West"], index=0)

q = st.text_area("Question", value="Which students haven’t submitted their homework yet?")
if st.button("Ask"):
    res = answer("data/students.csv", {"grade": grade, "class": clss, "region": region}, q)
    if "error" in res:
        st.error(res["error"])
        st.json(res)
    else:
        st.success(f"Intent: {res.get('intent')}")
        st.json(res)
