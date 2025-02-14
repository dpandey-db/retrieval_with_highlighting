import streamlit as st
import requests


def main():
    st.title("Reference Search")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        filter1 = st.selectbox("Category", ["All", "Category1", "Category2"])
    with col2:
        filter2 = st.multiselect("Types", ["Type1", "Type2", "Type3"])

    # Search input
    query = st.text_input("Search references")

    if st.button("Search"):
        # Call your endpoint
        response = requests.get(
            f"your_api_endpoint",
            params={"query": query, "filter1": filter1, "filter2": filter2},
        )
        results = response.json()

        # Display results
        for result in results:
            with st.expander(f"Reference: {result['title']}"):
                st.markdown(result["highlighted_text"])
                st.text(f"Source: {result['source']}")


if __name__ == "__main__":
    main()
