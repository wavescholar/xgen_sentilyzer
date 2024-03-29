"""Main module for the streamlit app"""
import streamlit as st

import os

cwd = os.getcwd()
print(cwd)

import main_page

from pages import page_2

from pages import page_3

breakpoint()

PAGES = {
    "Main": main_page,
    "P2": page_2,
    "P3": page_3,
}


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()

    st.sidebar.title("Contribute")
    st.sidebar.info("This an open source project ")
    st.sidebar.title("About")
    st.sidebar.info(
        """
        
"""
    )


if __name__ == "__main__":
    main()
