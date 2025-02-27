import streamlit as st
import pickle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load trained model
with open("transformer_model.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Streamlit UI
st.title("Pseudocode to C++ Code Generator")
st.write("Enter pseudocode below, and the model will generate equivalent C++ code.")

# Input field
pseudo_input = st.text_area("Enter Pseudocode:")

if st.button("Generate Code"):
    if pseudo_input.strip():
        # Tokenize input
        inputs = tokenizer(pseudo_input, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Generate output
        with torch.no_grad():
            output_ids = model.generate(inputs["input_ids"], max_length=128)

        # Decode output
        generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display output
        st.subheader("Generated C++ Code:")
        st.code(generated_code, language="cpp")
    else:
        st.error("Please enter some pseudocode.")

