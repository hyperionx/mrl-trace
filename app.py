from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import os
import time
import pandas as pd
from faker import Faker
import random
import streamlit as st

fake = Faker()

# Function to generate a random engagement level
def random_engagement_level():
    return random.choice(['Low', 'Medium', 'High'])

# Function to generate a random job title
def random_job_title():
    job_titles = ['CEO', 'CFO', 'CTO', 'COO', 'VP of Sales', 'Marketing Manager', 'Product Manager', 'Lead Engineer', 'Business Analyst']
    return random.choice(job_titles)

# Function to generate random client data
def generate_client_data(num_clients=500):
    clients = []
    
    for _ in range(num_clients):
        company_name = fake.company()
        num_key_individuals = random.randint(1, 3)  # Number of key contacts per company (1 to 3 individuals)

        for _ in range(num_key_individuals):
            client = {
                'Company Name': company_name,
                'Company LinkedIn': f"https://www.linkedin.com/company/{company_name.replace(' ', '').lower()}",
                'Company Website': fake.url(),
                'Deals Completed': random.randint(1, 50),
                'Processes Initiated': random.randint(0, 5),
                'Engagement Level': random_engagement_level(),
                'Key Individual Name': fake.name(),
                'Key Individual LinkedIn': f"https://www.linkedin.com/in/{fake.name().replace(' ', '').lower()}",
                'Email': fake.email(),
                'Phone Number': fake.phone_number(),
                'Job Title': random_job_title()  # Replace 'Status' with 'Job Title'
            }
            clients.append(client)

    return clients

# Generate the dummy client data
client_data = generate_client_data()

# Create a pandas DataFrame for better visualization and manipulation
df_clients = pd.DataFrame(client_data)

# Display the first few rows of the dataframe
print(df_clients.head())

# Save the data to a CSV file
# df_clients.to_csv('dummy_client_data.csv', index=False)

df_clients = pd.read_csv('dummy_client_data.csv')
# Extract the unique company names from the CSV
company_names = df_clients['Company Name'].unique()


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="""                        
Write a short summary without special characters, with the information on the hypothetical company in a random industry:

1. Add more detail to the recent deals and acquisitions in the industry, including any additional companies involved and any changes in deal amounts or nature.
2. Expand on the latest fundraising activities, highlighting any new investors or funding rounds that may have emerged after the last input.
3. Provide updates on new market entrants in the industry, focusing on any new trends or challenges.
4. Continue from the previous analysis of mandates, partnerships, or strategic alliances, and introduce any significant developments or changes in the industry.
                              
If contact information for the company is provided as part of the prompt, include the following:
- Key Individual: [Key Individual Name]
- Job Title: [Key Individual Job Title]
- Email: [Key Individual Email]
- Phone Number: [Key Individual Phone Number]

Ensure all information remains hypothetical and based on the current market trends in the specified industry.

Note: If the company’s name or sector is mentioned in the prompt, please ensure that contact information is added if it exists, in the format above. 
""")

# Function to generate response with contact information
def generate_response_from_gemini(messages, contact_info_list=None):
    # Convert messages into the correct format
    formatted_messages = [
            {
                "role": m["role"],
                "parts": [{"text": m["content"]}]
            } for m in messages
        ]

    if contact_info_list:
        # Add contact information if available
        contact_details = "### Contact Information:\n"
        for contact_info in contact_info_list:
            contact_details += f"""
            **Company Name:** {contact_info['Company Name']}
            - **Key Individual:** {contact_info['Key Individual']}
            - **Job Title:** {contact_info['Job Title']}
            - **Email:** {contact_info['Email']}
            - **Phone Number:** {contact_info['Phone Number']}
            """
        # Append the contact info to the messages
        formatted_messages[-1]["parts"].append({"text": contact_details})

    try:
        response = model.generate_content(
            formatted_messages,
            generation_config=genai.GenerationConfig(
                max_output_tokens=300,
                temperature=0.1,
            )
        )

        return response.text
    except Exception as e:
        return f"Error generating hypothetical public data: {e}"


# Function to retrieve all contact information for a given company
def get_contact_info(company_name):
    # Search the dataframe for all rows matching the company name (case-insensitive)
    company_rows = df_clients[df_clients['Company Name'].str.contains(company_name, case=False, na=False)]

    if not company_rows.empty:
        # Extract contact information for all rows matching the company name
        contact_info_list = []
        for index, row in company_rows.iterrows():
            contact_info = {
                'Company Name': row['Company Name'],
                'Key Individual': row['Key Individual Name'],
                'Job Title': row['Job Title'],
                'Email': row['Email'],
                'Phone Number': row['Phone Number']
            }
            contact_info_list.append(contact_info)
        return contact_info_list
    return None

# Streamlit UI layout
st.set_page_config(layout="wide")

def main():
    # Streamlit page setup
    st.title("Engagement Insight Generator")

    # Display the client data in a table
    st.subheader("Client Data Table")
    st.dataframe(df_clients)  # This will render the dataframe as an interactive table

    # Initialize the chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Move the chatbot functionality into the sidebar
    with st.sidebar:
        st.header("Chat with the Assistant")

        # Input box for the user (always stays at the bottom)
        prompt = st.text_input("Ask a question or provide a prompt:")

        if prompt:
            # Append the user's message to the chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Check if the user is asking about a company in the dataframe
            contact_info_list = None
            for company_name in company_names:
                if company_name.lower() in prompt.lower():  # If the company is mentioned in the prompt
                    contact_info_list = get_contact_info(company_name)
                    if contact_info_list:
                        break  # Exit the loop if contact info is found

            # Generate the assistant's response based on the conversation history
            response = generate_response_from_gemini(
                [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                contact_info_list=contact_info_list
            )

            # Display the assistant's response in the chat
            with st.chat_message("assistant"):
                st.markdown(response)

            # Append the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()