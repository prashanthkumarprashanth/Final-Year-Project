import streamlit as st
import webbrowser

def main():
    st.title("LinkedIn Profile Redirector")
    st.write("Click the button below to visit Mohan Sai Dinesh Boddapati's LinkedIn profile.")

    if st.button("Submit"):
        linkedin_url = "https://travel-g4hrr4h68ds5xh3q9jgzzx.streamlit.app/"
        st.success("Redirecting to LinkedIn profile...")
        webbrowser.open_new_tab(linkedin_url)

if __name__ == "__main__":
    main()
