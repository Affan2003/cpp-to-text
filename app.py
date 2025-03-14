import streamlit as st
import torch
import sentencepiece as spm

# Load Tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# Load Model
model = torch.load("transformer_model.pth", map_location=torch.device('cpu'))
model.eval()

def generate_text(cpp_code):
    """Generate textual description from C++ code."""
    input_tokens = sp.encode(cpp_code, out_type=int)
    input_tensor = torch.tensor([input_tokens])
    with torch.no_grad():
        output_tokens = model(input_tensor)
    output_text = sp.decode(output_tokens.argmax(dim=-1).squeeze().tolist())
    return output_text

# Streamlit UI
st.title("C++ Code to Text Description Generator")
st.write("Enter C++ code and get the corresponding textual description.")

user_input = st.text_area("Enter C++ Code:")
if st.button("Generate Description"):
    if user_input.strip():
        output_text = generate_text(user_input)
        st.write("### Generated Description:")
        st.write(output_text)
    else:
        st.warning("Please enter some C++ code!")
