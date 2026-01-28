import os
import torch
import base64
import io
import datetime
import logging
from flask import Flask, request, render_template, jsonify, send_file
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from fpdf import FPDF

# Suppress Hugging Face warnings and model loading reports
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- 1. Initialize Application and Model ---
app = Flask(__name__)

MODEL_PATH = "potato_disease_model.pth"
# Replace YOUR_GOOGLE_DRIVE_FILE_ID with the actual ID from your Google Drive link
MODEL_URL = "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_FILE_ID"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        import urllib.request
        import ssl
        # Create SSL context to handle HTTPS
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check the Google Drive link and ensure it's publicly accessible")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'. Please upload to Google Drive and update MODEL_URL.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=3)
# Load the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)
model.eval()

# Load class names from the checkpoint
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load the image processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# --- 2. Disease Information ---
disease_info = {
    "Potato___Early_blight": {
        "description": "A common fungal disease caused by Alternaria solani.",
        "symptoms": "Small, dark, circular spots with 'bull's-eye' rings on lower leaves. Yellow halo around spots. Leaves may wither and die.",
        "action": "Apply fungicides, practice crop rotation, remove infected plant debris, and ensure proper air circulation."
    },
    "Potato___Late_blight": {
        "description": "A destructive fungal disease caused by Phytophthora infestans.",
        "symptoms": "Large, dark, water-soaked spots on leaves, often with a white mold on the underside. Can spread rapidly and destroy entire crops.",
        "action": "Apply protective fungicides preventatively. Remove and destroy infected plants immediately. Ensure good drainage and air flow."
    },
    "Potato___healthy": {
        "description": "The plant appears to be free of common fungal diseases.",
        "symptoms": "Leaves are uniformly green, with no significant spots, yellowing, or signs of decay.",
        "action": "Continue good plant care practices, monitor regularly for signs of stress or disease, and ensure proper nutrition and watering."
    }
}

# --- 3. Prediction Logic ---
def predict_single_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = idx_to_class[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item()
    
    formatted_name = predicted_class_name.replace("___", " ").replace("_", " ")

    buffered_original = io.BytesIO()
    image.save(buffered_original, format="JPEG")
    original_img_str = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

    return {
        'prediction': formatted_name,
        'confidence': f"{(confidence * 100):.2f}%",
        'info': disease_info.get(predicted_class_name, {}),
        'image': f"data:image/jpeg;base64,{original_img_str}"
    }

# --- 4. Detailed PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Potato Leaf Disease Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 9)
        self.cell(0, 8, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def result_entry(self, entry_data):
        try:
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(230, 230, 230)
            
            # Clean filename for PDF
            filename = str(entry_data.get('filename', 'Unknown')).encode('latin-1', 'replace').decode('latin-1')
            self.cell(0, 10, f"File: {filename}", 1, 1, 'L', 1)
            self.ln(4)

            self.set_font('Arial', 'B', 10)
            self.cell(35, 8, 'Prediction:', 0, 0)
            self.set_font('Arial', '', 10)
            
            # Clean prediction text
            prediction = str(entry_data.get('prediction', 'N/A')).encode('latin-1', 'replace').decode('latin-1')
            self.cell(0, 8, prediction, 0, 1)

            self.set_font('Arial', 'B', 10)
            self.cell(35, 8, 'Confidence:', 0, 0)
            self.set_font('Arial', '', 10)
            confidence = str(entry_data.get('confidence', 'N/A'))
            self.cell(0, 8, confidence, 0, 1)
            self.ln(2)

            if entry_data.get('info'):
                info = entry_data['info']
                self.set_font('Arial', 'B', 10)
                self.cell(0, 8, 'Details', 0, 1)
                self.set_font('Arial', '', 10)
                
                # Clean text for PDF encoding
                description = str(info.get('description', 'N/A')).encode('latin-1', 'replace').decode('latin-1')
                symptoms = str(info.get('symptoms', 'N/A')).encode('latin-1', 'replace').decode('latin-1')
                action = str(info.get('action', 'N/A')).encode('latin-1', 'replace').decode('latin-1')
                
                self.cell(5)  # Indent
                self.multi_cell(0, 5, f"Description: {description}")
                self.cell(5)
                self.multi_cell(0, 5, f"Symptoms: {symptoms}")
                self.cell(5)
                self.multi_cell(0, 5, f"Recommended Action: {action}")
            
            self.ln(10)
        except Exception as e:
            print(f"Error adding entry to PDF: {e}")
            # Add a simple error entry
            self.set_font('Arial', '', 10)
            self.cell(0, 8, f"Error processing entry: {entry_data.get('filename', 'Unknown')}", 0, 1)
            self.ln(5)
        self.ln(10)


# --- 5. Flask Routes ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_batch():
    if 'files[]' not in request.files: return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '': return jsonify({'error': 'No selected files'}), 400

    results = []
    for file in files:
        try:
            result = predict_single_image(file.read())
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({'filename': file.filename, 'error': 'Could not process file.'})
            
    return jsonify(results)

@app.route('/export', methods=['POST'])
def export_report():
    data = request.json.get('data', [])
    file_format = request.json.get('format')

    if file_format == 'pdf':
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create the PDF document
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph("Potato Leaf Disease Analysis Report", title_style))
            story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Process each result
            for i, entry in enumerate(data):
                if 'error' not in entry:
                    # Clean filename - remove hash part if present
                    original_filename = entry.get('filename', 'Unknown')
                    # Handle different filename formats with hash prefixes
                    if '___' in original_filename:
                        # Split by ___ and take the last part (actual filename)
                        clean_filename = original_filename.split('___')[-1]
                    elif '-' in original_filename and len(original_filename.split('-')[0]) > 10:
                        # Handle format like "00d8f10f-5038-4e0f-bb58-0b885ddc0cc5___RS_Early.B_8722.JPG"
                        # Find the part after the last long hash segment
                        parts = original_filename.split('-')
                        if len(parts) > 4:  # Likely has a UUID-style hash
                            # Look for the actual filename part (usually after ___)
                            for part in parts:
                                if '___' in part:
                                    clean_filename = part.split('___')[-1]
                                    break
                            else:
                                # If no ___ found, take the last meaningful part
                                clean_filename = parts[-1] if parts[-1] else original_filename
                        else:
                            clean_filename = original_filename
                    else:
                        clean_filename = original_filename
                    
                    # Additional cleanup - remove any remaining hash-like prefixes
                    if clean_filename.count('_') > 3 and len(clean_filename) > 30:
                        # Look for pattern like "hash___actualname"
                        if '___' in clean_filename:
                            clean_filename = clean_filename.split('___')[-1]
                    
                    # Create descriptive filename based on prediction
                    prediction = entry.get('prediction', '').lower()
                    
                    if 'healthy' in prediction:
                        display_filename = f"Healthy_Leaf_{i+1}.jpg"
                    elif 'early' in prediction and 'blight' in prediction:
                        display_filename = f"Early_Blight_{i+1}.jpg"
                    elif 'late' in prediction and 'blight' in prediction:
                        display_filename = f"Late_Blight_{i+1}.jpg"
                    else:
                        # Fallback to cleaned original filename
                        display_filename = clean_filename
                    
                    # File header
                    story.append(Paragraph(f"<b>Analysis {i+1}: {display_filename}</b>", styles['Heading2']))
                    
                    # Basic info table (only prediction and confidence)
                    basic_data = [
                        ['Prediction:', entry.get('prediction', 'N/A')],
                        ['Confidence:', entry.get('confidence', 'N/A')]
                    ]
                    
                    basic_table = Table(basic_data, colWidths=[1.5*inch, 4.5*inch])
                    basic_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 11),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    
                    story.append(basic_table)
                    story.append(Spacer(1, 15))
                    
                    # Detailed information as separate paragraphs
                    if entry.get('info'):
                        info = entry['info']
                        
                        # Create a custom style for section headers
                        section_style = ParagraphStyle(
                            'SectionHeader',
                            parent=styles['Heading3'],
                            fontSize=12,
                            spaceAfter=5,
                            textColor=colors.darkgreen,
                            fontName='Helvetica-Bold'
                        )
                        
                        # Description
                        story.append(Paragraph("Description:", section_style))
                        story.append(Paragraph(info.get('description', 'N/A'), styles['Normal']))
                        story.append(Spacer(1, 10))
                        
                        # Symptoms
                        story.append(Paragraph("Symptoms:", section_style))
                        story.append(Paragraph(info.get('symptoms', 'N/A'), styles['Normal']))
                        story.append(Spacer(1, 10))
                        
                        # Recommended Action
                        story.append(Paragraph("Recommended Action:", section_style))
                        story.append(Paragraph(info.get('action', 'N/A'), styles['Normal']))
                        story.append(Spacer(1, 20))
                    
                    # Add separator line between analyses
                    if i < len([x for x in data if 'error' not in x]) - 1:
                        story.append(Spacer(1, 10))
                        separator_style = ParagraphStyle(
                            'Separator',
                            parent=styles['Normal'],
                            alignment=1,
                            textColor=colors.lightgrey
                        )
                        story.append(Paragraph("─" * 80, separator_style))
                        story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name='potato_disease_report.pdf',
                mimetype='application/pdf'
            )
            
        except ImportError:
            # Fallback to fpdf2 if reportlab is not available
            try:
                pdf = PDF()
                pdf.add_page()
                for entry in data:
                    if 'error' not in entry:
                        pdf.result_entry(entry)
                
                buffer = io.BytesIO()
                pdf_bytes = pdf.output()
                buffer.write(pdf_bytes)
                buffer.seek(0)
                
                return send_file(
                    buffer,
                    as_attachment=True,
                    download_name='potato_disease_report.pdf',
                    mimetype='application/pdf'
                )
            except Exception as e:
                print(f"PDF generation error: {e}")
                return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
                
        except Exception as e:
            print(f"PDF generation error: {e}")
            return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
    
    elif file_format == 'csv':
        try:
            output = io.StringIO()
            output.write("Filename,Prediction,Confidence,Description,Symptoms,Action\n")
            
            for row in data:
                if 'error' not in row:
                    info = row.get('info', {})
                    # Escape quotes and handle special characters
                    filename = str(row.get('filename', '')).replace('"', '""')
                    prediction = str(row.get('prediction', '')).replace('"', '""')
                    confidence = str(row.get('confidence', '')).replace('"', '""')
                    description = str(info.get('description', '')).replace('"', '""')
                    symptoms = str(info.get('symptoms', '')).replace('"', '""')
                    action = str(info.get('action', '')).replace('"', '""')
                    
                    output.write(f'"{filename}","{prediction}","{confidence}","{description}","{symptoms}","{action}"\n')
            
            buffer = io.BytesIO()
            buffer.write(output.getvalue().encode('utf-8'))
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name='potato_disease_report.csv',
                mimetype='text/csv'
            )
        except Exception as e:
            print(f"CSV generation error: {e}")
            return jsonify({'error': f'CSV generation failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid format'}), 400

# --- 6. Run Application ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

