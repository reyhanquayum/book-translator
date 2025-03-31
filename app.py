import os
from flask import (
    Flask,
    request,
    render_template,
    send_from_directory,
    redirect,
    url_for,
    flash,
)
import time  # Added for Gemini file processing check
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import google.generativeai as genai  # Added for Gemini
from dotenv import load_dotenv # Added to load .env file

load_dotenv() # Load environment variables from .env file

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load from .env
UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "downloads"
ALLOWED_EXTENSIONS = {"pdf"}

# Check if GEMINI_API_KEY is loaded and configure Gemini
gemini_configured = False
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file or environment variables. Gemini OCR will be disabled.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        print("Gemini configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini with loaded API key: {e}. Gemini OCR will be disabled.")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
# It's important to set a secret key for flashing messages
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "default-secret-key-for-dev") # Load secret key or use default


# Ensure upload and download directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["DOWNLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Renders the main upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads."""
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file.", "error")
        return redirect(url_for("index"))

    if file and allowed_file(file.filename):
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(filepath)
            flash(f'File "{filename}" uploaded successfully.', "info")

            # --- Text Extraction (Refactored for Page-by-Page Processing) ---
            pymupdf_text = ""
            gemini_ocr_text_parts = [] # List to hold OCR text for each page
            gemini_translated_text_parts = [] # List to hold translated text for each page
            final_text = ""
            extraction_method = "N/A"
            text_filename = filename.rsplit(".", 1)[0] + ".txt"
            text_filepath = os.path.join(app.config["DOWNLOAD_FOLDER"], text_filename)
            total_pages = 0

            try:
                flash('Starting text extraction process...', 'info')
                doc = None # Initialize doc to None

                # --- Attempt 1: PyMuPDF Raw Text Extraction (as fallback) ---
                flash('Attempting raw text extraction with PyMuPDF (as fallback)...', 'info')
                try:
                    doc = fitz.open(filepath)
                    total_pages = len(doc)
                    for page_num in range(total_pages):
                        page = doc.load_page(page_num)
                        pymupdf_text += page.get_text()
                    if pymupdf_text.strip():
                        flash(f'PyMuPDF raw extraction completed ({total_pages} pages).', 'info')
                    else:
                        flash('PyMuPDF extracted no raw text.', 'warning')
                except Exception as pymu_error:
                    flash(f"Error during PyMuPDF raw extraction: {str(pymu_error)}", "error")
                    pymupdf_text = "" # Ensure empty on error
                finally:
                    if doc:
                        doc.close() # Ensure document is closed

                # --- Attempt 2: Gemini Page-by-Page OCR + Translation ---
                if gemini_configured:
                    flash("Attempting Page-by-Page OCR & Translation with Gemini...", 'info')
                    doc = None # Re-initialize doc
                    try:
                        doc = fitz.open(filepath)
                        total_pages = len(doc)
                        flash(f"Processing {total_pages} pages...", 'info')

                        # Define models outside the loop
                        # Use the specific experimental model for OCR (as originally intended, but applied page-by-page)
                        # Note: This requires the image to be passed differently than to gemini-pro-vision
                        ocr_model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
                        # Use the standard text model for translation
                        translation_model = genai.GenerativeModel("gemini-pro")

                        # OCR prompt needs to be adjusted as this model doesn't take image bytes directly in the same way
                        # We might need to revert to uploading the file and referencing it if this model doesn't support inline images.
                        # For now, let's assume it might work with a standard text prompt if the context is right,
                        # but this is less likely for OCR. Let's adjust the prompt first.
                        # **Correction**: The experimental text model likely CANNOT process images directly like gemini-pro-vision.
                        # The page-by-page image approach requires gemini-pro-vision.
                        # Let's stick with gemini-pro-vision for OCR and gemini-pro for translation as the most likely working combination.
                        # Reverting the model change I was about to make.

                        # Re-confirming the models based on user feedback and deprecation notice:
                        # Use gemini-1.5-flash-latest for OCR as it's multimodal and replaces gemini-pro-vision
                        ocr_model = genai.GenerativeModel("gemini-1.5-flash-latest")
                        # Try gemini-1.5-pro-latest for translation as gemini-pro was not found
                        translation_model = genai.GenerativeModel("gemini-1.5-pro-latest")

                        ocr_prompt_template = """Extract all text content from the provided single page image.
Format the output exactly like this:

==Start of OCR for page {page_number}==
[Text content of the page]
==End of OCR for page {page_number}==

- Extract the text content as accurately as possible between the markers.
"""
                        translation_prompt_template = """You are an expert translator specializing in Islamic jurisprudence texts from Urdu to English.
Translate the following text, which is OCR output from a single page of the book 'Asbab Ikhtilaf al-Fuqaha'. Adhere strictly to these rules:
1.  Translate **only the Urdu** portions into clear, accurate, and formal English suitable for the subject matter.
2.  Leave **all Arabic text** (like Qur'anic verses, Hadith snippets, standard Arabic phrases) **exactly as it is** in the original Arabic script. Do not translate or transliterate it.
3.  **Transliterate** common Islamic/fiqhi terms using a consistent scheme (e.g., Fiqh, Hadith, Sahabi, Imam, Sanad, Usul, Shar‘ī, Sunnah, Qur'an, ‘Alim, ‘Ulama’, Taqlid, Ijtihad, Fatwa, Halal, Haram). Do not translate these specific terms into English words like 'jurisprudence' or 'tradition'.
4.  Maintain the page structure indicated by the `==Start/End of OCR for page X==` markers. Translate the content between the markers.
5.  If you encounter ambiguous phrases or potential OCR errors in the Urdu, translate as best as possible and optionally add a brief translator note like `[TN: Possible OCR error for 'word']` or `[TN: Phrase interpretation...]`.

Input Text (Single Page):
{ocr_text}
"""

                        for page_num in range(total_pages):
                            current_page_index = page_num + 1 # User-friendly page number (1-based)
                            flash(f"Processing Page {current_page_index}/{total_pages}...", 'info')
                            page_ocr_text = ""
                            page_translated_text = ""

                            try:
                                page = doc.load_page(page_num)

                                # --- Step 2a: OCR Page with Gemini Vision ---
                                flash(f"  - Step 1/2: OCR Page {current_page_index}", 'info')
                                # Render page to an image (pixmap)
                                pix = page.get_pixmap(dpi=300) # Higher DPI for better OCR
                                img_bytes = pix.tobytes("png") # Convert to PNG bytes

                                # Prepare image part for Gemini API
                                image_part = {"mime_type": "image/png", "data": img_bytes}
                                ocr_prompt = ocr_prompt_template.format(page_number=current_page_index)

                                # Send image and prompt to Gemini Vision
                                try:
                                    ocr_response = ocr_model.generate_content(
                                        [image_part, ocr_prompt],
                                        request_options={'timeout': 180} # 3 min timeout per page OCR
                                    )
                                except Exception as ocr_api_error:
                                     # Catch API errors specifically during the call
                                     flash(f"  - Step 1/2: API Error during OCR for Page {current_page_index}: {str(ocr_api_error)}", "error")
                                     ocr_response = None # Ensure response is None on API error

                                if ocr_response and ocr_response.text:
                                    page_ocr_text = ocr_response.text
                                    gemini_ocr_text_parts.append(page_ocr_text) # Store raw OCR for potential fallback
                                    flash(f"  - Step 1/2: OCR Page {current_page_index} successful.", 'info')
                                else:
                                    flash(f"  - Step 1/2: OCR Page {current_page_index} returned no text.", 'warning')
                                    # Append placeholder or empty string? Let's append empty for now.
                                    gemini_ocr_text_parts.append("")


                                # --- Step 2b: Translate Page (if OCR successful) ---
                                if page_ocr_text:
                                    flash(f"  - Step 2/2: Translating Page {current_page_index}", 'info')
                                    try:
                                        translation_prompt = translation_prompt_template.format(ocr_text=page_ocr_text)
                                        try:
                                            translation_response = translation_model.generate_content(
                                                translation_prompt,
                                                request_options={'timeout': 240} # 4 min timeout per page translation
                                            )
                                        except Exception as trans_api_error:
                                            # Catch API errors specifically during the call
                                            flash(f"  - Step 2/2: API Error during Translation for Page {current_page_index}: {str(trans_api_error)}", "error")
                                            translation_response = None # Ensure response is None on API error


                                        if translation_response and translation_response.text:
                                            page_translated_text = translation_response.text
                                            gemini_translated_text_parts.append(page_translated_text)
                                            flash(f"  - Step 2/2: Translation Page {current_page_index} successful.", 'info')
                                        else:
                                            flash(f"  - Step 2/2: Translation Page {current_page_index} returned no text.", 'warning')
                                            gemini_translated_text_parts.append("") # Append empty if translation fails
                                    except Exception as translate_error:
                                        flash(f"  - Step 2/2: Error translating Page {current_page_index}: {str(translate_error)}", "error")
                                        gemini_translated_text_parts.append("") # Append empty on error
                                else:
                                     # If OCR failed, we can't translate, append empty string
                                     gemini_translated_text_parts.append("")
                                     flash(f"  - Step 2/2: Skipping translation for Page {current_page_index} due to OCR failure.", 'warning')

                            except Exception as page_error:
                                flash(f"Error processing Page {current_page_index}: {str(page_error)}", "error")
                                # Append empty strings if page processing fails
                                gemini_ocr_text_parts.append("")
                                gemini_translated_text_parts.append("")
                                # Continue to the next page

                        flash("Gemini page-by-page processing finished.", 'info')

                    except Exception as gemini_loop_error:
                        flash(f'Error during Gemini page processing loop: {str(gemini_loop_error)}', "error")
                        # Clear partial results if the loop fails catastrophically? Or keep them? Let's clear.
                        gemini_ocr_text_parts = []
                        gemini_translated_text_parts = []
                    finally:
                        if doc:
                            doc.close() # Ensure document is closed

                else:
                    flash("Gemini OCR & Translation skipped: API key not configured.", "warning")


                # --- Decide which text to use and Save ---
                flash("Determining final text source...", 'info')
                # Combine the translated parts if available
                combined_gemini_text = "\n\n".join(part for part in gemini_translated_text_parts if part)

                if combined_gemini_text.strip():
                    final_text = combined_gemini_text
                    extraction_method = "Gemini (Page-by-Page OCR + Translation)"
                    flash(f"Prioritizing Gemini page-by-page result ({len(gemini_translated_text_parts)} pages processed).", 'info')
                elif pymupdf_text.strip():
                    final_text = pymupdf_text
                    extraction_method = "PyMuPDF (Raw Text)"
                    flash("Gemini process failed or yielded no text; falling back to PyMuPDF raw text result.", 'warning')
                else:
                    extraction_method = "None"
                    flash("Both PyMuPDF and Gemini processes failed to extract text.", 'error')

                # Save the final extracted text if any
                if final_text.strip():
                    flash(f"Saving extracted text ({extraction_method}) to '{text_filename}'...", 'info')
                    with open(text_filepath, "w", encoding="utf-8") as text_file:
                        text_file.write(final_text)
                    flash(f'Text successfully saved to "{text_filename}".', 'success')
                else:
                    flash(f'No text could be extracted to save for "{filename}".', 'warning')

            except Exception as processing_error:
                # Catch any unexpected error during the whole process
                flash(f'Unexpected error during processing of "{filename}": {str(processing_error)}', "error")
                # Ensure doc is closed if an error happens early
                if 'doc' in locals() and doc and not doc.is_closed:
                    doc.close()


            # --- End Text Extraction ---

            # Redirect back to index regardless of extraction success/failure for now
            return redirect(url_for("index"))
        except Exception as save_error:
            flash(
                f"An error occurred while saving the file: {str(save_error)}", "error"
            )
            return redirect(url_for("index"))
    elif file and not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PDF.", "error")
        return redirect(url_for("index"))
    else:
        # This case should theoretically not be reached due to prior checks
        flash("An unexpected error occurred during upload.", "error")
        return redirect(url_for("index"))


@app.route("/download/<filename>")
def download_file(filename):
    """Serves files from the download directory."""
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        flash("Invalid filename for download.", "error")
        return redirect(url_for("index"))

    try:
        return send_from_directory(
            app.config["DOWNLOAD_FOLDER"],
            safe_filename,
            as_attachment=True,  # Force download prompt
        )
    except FileNotFoundError:
        flash(f'File "{safe_filename}" not found in download area.', "error")
        return redirect(url_for("index"))
    except Exception as e:
        app.logger.error(
            f"Error downloading file {safe_filename}: {e}"
        )
        flash("An error occurred while trying to download the file.", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
