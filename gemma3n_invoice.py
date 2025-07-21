#%%
# Invoice OCR and Data Structuring using Gemma 3n Vision Model

!pip uninstall -y pygobject gradient gradient-utils
!pip install --no-cache-dir --upgrade packaging==24.2
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install --upgrade bitsandbytes
!pip install triton==3.2.0
!pip install pip3-autoremove
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth
!pip install faiss-cpu
!pip install sentence-transformers
!pip install wikipedia
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm

#%%
import os
import json
import glob
from typing import List, Dict, Any, Optional
from unsloth import FastModel
from PIL import Image
from IPython.display import display, Image as IPImage

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

#%%
# Load the model (exactly as in CLAUDE.md)
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None,
    max_seq_length = 2048,
    load_in_4bit = True,
    full_finetuning = False,
)

#%%
# Data loading for invoice images
base_dir = '/notebooks/kaggle/input/Invoice_Images'

# Get all invoice image files (JPG, PNG)
def load_invoice_images(directory):
    invoice_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        invoice_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return invoice_files

invoice_files = load_invoice_images(base_dir)
print(f"Found {len(invoice_files)} invoice images")

# Create dataset structure for invoice processing
invoice_dataset = []
for file_path in invoice_files:
    invoice_dataset.append({
        "path": file_path,
        "filename": os.path.basename(file_path)
    })

#%%
# Create invoice OCR instruction with structured JSON output format
instruction = """You are an expert invoice OCR system. Extract all relevant information from this invoice image and return it as a structured JSON object.

Return ONLY a valid JSON object with the following exact structure. Use empty strings ("") for missing information. Do not wrap the JSON in markdown code blocks or any other formatting:

{
  "invoice": {
    "client_name": "Client company name",
    "client_address": "Client full address",
    "seller_name": "Seller/vendor company name", 
    "seller_address": "Seller full address",
    "invoice_number": "Invoice number",
    "invoice_date": "Invoice date (MM/DD/YYYY format)",
    "due_date": "Due date if available"
  },
  "items": [
    {
      "description": "Item description",
      "quantity": "Quantity as string",
      "total_price": "Total price as string"
    }
  ],
  "subtotal": {
    "tax": "Tax amount as string",
    "discount": "Discount amount as string", 
    "total": "Final total amount as string"
  },
  "payment_instructions": {
    "due_date": "Payment due date",
    "bank_name": "Bank name if available",
    "account_number": "Account number if available",
    "payment_method": "Payment method if available"
  }
}

Extract all line items from the invoice. For addresses, include complete address with line breaks (\\n) for multi-line addresses. All numeric values should be extracted as strings to preserve formatting."""

#%%
# Helper function for invoice OCR inference
def do_invoice_ocr(image_path, max_new_tokens=1024):
    try:
        # Load and process image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]
        }]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Try to parse as JSON - handle markdown code blocks
        try:
            # Clean the response text - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            
            # Check if response is wrapped in markdown code blocks
            if cleaned_text.startswith("```json"):
                # Extract JSON content after ```json
                json_start = cleaned_text.find("```json") + 7
                # Look for closing ``` or use rest of text if truncated
                json_end = cleaned_text.rfind("```")
                if json_end > json_start:  # Found closing ```
                    cleaned_text = cleaned_text[json_start:json_end].strip()
                else:  # No closing ``` (truncated), use everything after ```json
                    cleaned_text = cleaned_text[json_start:].strip()
            elif cleaned_text.startswith("```"):
                # Handle generic code blocks
                lines = cleaned_text.split('\n')
                if len(lines) > 2:
                    # Look for closing ``` or use available lines if truncated
                    if cleaned_text.endswith("```"):
                        cleaned_text = '\n'.join(lines[1:-1]).strip()
                    else:
                        cleaned_text = '\n'.join(lines[1:]).strip()
            
            structured_data = json.loads(cleaned_text)
            return structured_data
            
        except json.JSONDecodeError as e:
            print("⚠️ Failed to parse response as JSON")
            print(f"JSON Error: {e}")
            print(f"Cleaned text preview: {cleaned_text[:200]}...")
            return {"raw_response": response_text, "cleaned_response": cleaned_text, "error": f"Invalid JSON format: {e}"}
            
    except Exception as e:
        print(f"❌ Error processing invoice: {e}")
        return {"error": str(e)}

#%%
# Data validation function for new JSON structure
def validate_invoice_data(data):
    validation_results = {
        'is_valid': True,
        'missing_fields': [],
        'warnings': []
    }
    
    # Check main structure
    required_sections = ['invoice', 'items', 'subtotal', 'payment_instructions']
    for section in required_sections:
        if section not in data:
            validation_results['missing_fields'].append(section)
            validation_results['is_valid'] = False
    
    # Check invoice section
    if 'invoice' in data:
        invoice_required = ['seller_name', 'invoice_number']
        for field in invoice_required:
            if field not in data['invoice'] or not data['invoice'][field]:
                validation_results['missing_fields'].append(f"invoice.{field}")
                validation_results['is_valid'] = False
    
    # Check items array
    if 'items' in data:
        if not isinstance(data['items'], list) or len(data['items']) == 0:
            validation_results['warnings'].append('No items found in invoice')
        else:
            for i, item in enumerate(data['items']):
                if 'description' not in item or not item['description']:
                    validation_results['warnings'].append(f'Item {i+1} missing description')
    
    # Check subtotal
    if 'subtotal' in data:
        if 'total' not in data['subtotal'] or not data['subtotal']['total']:
            validation_results['warnings'].append('Missing total amount')
    
    return validation_results

#%%
# Batch processing function for multiple invoices
def process_invoices_batch(invoice_paths, max_files=None):
    results = []
    files_to_process = invoice_paths[:max_files] if max_files else invoice_paths
    
    for i, invoice_path in enumerate(files_to_process):
        print(f"Processing {i+1}/{len(files_to_process)}: {os.path.basename(invoice_path)}")
        
        result = do_invoice_ocr(invoice_path)
        result['filename'] = os.path.basename(invoice_path)
        result['file_path'] = invoice_path
        
        if 'error' not in result:
            validation = validate_invoice_data(result)
            result['validation'] = validation
        
        results.append(result)
    
    return results

#%%
# Test the model on a sample invoice
if invoice_dataset:
    print("\n🧪 Testing invoice OCR...")
    test_sample = invoice_dataset[0]
    
    print(f"\n🖼️ Processing: {test_sample['filename']}")
    print("=" * 50)
    
    # Test inference
    result = do_invoice_ocr(test_sample['path'])
    
    if 'error' not in result:
        print("✅ Extraction successful!")
        print(json.dumps(result, indent=2))
        
        # Validate extracted data
        validation = validate_invoice_data(result)
        print(f"\n📋 Validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
        if validation['missing_fields']:
            print(f"Missing fields: {validation['missing_fields']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
    else:
        print(f"❌ Extraction failed: {result}")

#%%
# Process multiple invoices (optional)
print("\n📄 Processing multiple invoices...")
batch_results = process_invoices_batch(invoice_files, max_files=5)

# Save results to JSON file
output_file = "invoice_extraction_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(batch_results, f, indent=2, ensure_ascii=False)

print(f"✅ Results saved to {output_file}")

# Display summary
successful_extractions = sum(1 for r in batch_results if 'error' not in r)
print(f"\n📊 Summary:")
print(f"Total processed: {len(batch_results)}")
print(f"Successful extractions: {successful_extractions}")
print(f"Failed extractions: {len(batch_results) - successful_extractions}")
# %%
