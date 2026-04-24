import sys
import os

# Add the parent directory (project root) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from orchestrator.router import route

st.title("FPL-Intel")

query = st.text_input("Ask something")

if query:
    result = route(query)
    st.write(result)
