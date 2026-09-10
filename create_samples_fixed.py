"""Create sample expected.pdf and actual.pdf for testing."""
import pymupdf  # PyMuPDF

def create_expected_pdf():
    """Create the expected PDF with baseline content."""
    doc = pymupdf.open()
    
    # Page 1: Title and content
    page = doc.new_page(width=612, height=792)
    
    # Header
    page.insert_text((50, 50), "PDF Validation Test Report", fontsize=24, color=(0, 0, 0))
    page.insert_text((50, 80), "Expected Document", fontsize=12, color=(0.4, 0.4, 0.4))
    
    # Content section
    page.insert_text((50, 120), "Introduction", fontsize=14)
    text = """This is a test document for PDF validation.
It contains multiple content types that will be compared.
The validation engine checks for differences in:
• Text content and position
• Images and their properties
• Tables and cell layouts
• Overall visual appearance"""
    y = 140
    for line in text.split('\n'):
        page.insert_text((50, y), line, fontsize=10)
        y += 15
    
    # Insert a test image (blue rectangle)
    img_rect = pymupdf.Rect(50, 320, 200, 420)
    page.draw_rect(img_rect, color=(0, 0, 1), fill=(0.4, 0.6, 1))
    page.insert_text((50, 440), "Figure 1: Test Image", fontsize=10)
    
    # Insert a table
    rows = [
        ["Column A", "Column B", "Column C"],
        ["Data 1.1", "Data 1.2", "Data 1.3"],
        ["Data 2.1", "Data 2.2", "Data 2.3"],
        ["Data 3.1", "Data 3.2", "Data 3.3"],
    ]
    
    table_rect = pymupdf.Rect(50, 480, 500, 600)
    x, y = 50, 480
    row_height = 30
    col_width = 150
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_rect = pymupdf.Rect(x + j * col_width, y + i * row_height, 
                                 x + (j + 1) * col_width, y + (i + 1) * row_height)
            page.draw_rect(cell_rect, color=(0, 0, 0))
            page.insert_text((x + j * col_width + 5, y + i * row_height + 10), 
                           cell, fontsize=9)
    
    page.insert_text((50, 650), "Page 1 of 2", fontsize=10, color=(0.5, 0.5, 0.5))
    
    # Page 2: Additional content
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 50), "Page 2: More Content", fontsize=20)
    page.insert_text((50, 100), "Additional section with more details", fontsize=12)
    
    content = """This section contains supplementary information.
It includes various content types to test the validation engine.
Different text sizes and styles are used throughout."""
    
    y = 140
    for line in content.split('\n'):
        page.insert_text((50, y), line, fontsize=10)
        y += 20
    
    # Add footer to page 2
    page.insert_text((50, 750), "Page 2 of 2", fontsize=10, color=(0.5, 0.5, 0.5))
    
    doc.save("sample/expected.pdf")
    doc.close()

def create_actual_pdf():
    """Create the actual PDF with minor differences for comparison."""
    doc = pymupdf.open()
    
    # Page 1: Title and content (with differences)
    page = doc.new_page(width=612, height=792)
    
    # Header
    page.insert_text((50, 50), "PDF Validation Test Report", fontsize=24, color=(0, 0, 0))
    page.insert_text((50, 80), "Actual Document", fontsize=12, color=(0.4, 0.4, 0.4))  # Different text
    
    # Content section
    page.insert_text((50, 120), "Introduction", fontsize=14)
    text = """This is a test document for PDF validation.
It contains multiple content types that will be compared.
The validation engine checks for differences in:
• Text content and position
• Images and their properties
• Tables and cell layouts
• Overall visual appearance"""
    y = 140
    for line in text.split('\n'):
        page.insert_text((50, y), line, fontsize=10)
        y += 15
    
    # Insert a test image (slightly different - red rectangle)
    img_rect = pymupdf.Rect(50, 320, 210, 420)  # Slightly wider
    page.draw_rect(img_rect, color=(1, 0, 0), fill=(1, 0.4, 0.4))  # Red instead of blue
    page.insert_text((50, 440), "Figure 1: Test Image", fontsize=10)
    
    # Insert a table (with different data)
    rows = [
        ["Column A", "Column B", "Column C"],
        ["Data 1.1", "Data 1.2 Modified", "Data 1.3"],  # Modified
        ["Data 2.1", "Data 2.2", "Data 2.3"],
        ["Data 3.1", "Data 3.2", "Data 3.3"],
    ]
    
    table_rect = pymupdf.Rect(50, 480, 500, 600)
    x, y = 50, 480
    row_height = 30
    col_width = 150
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_rect = pymupdf.Rect(x + j * col_width, y + i * row_height, 
                                 x + (j + 1) * col_width, y + (i + 1) * row_height)
            page.draw_rect(cell_rect, color=(0, 0, 0))
            page.insert_text((x + j * col_width + 5, y + i * row_height + 10), 
                           cell, fontsize=9)
    
    page.insert_text((50, 650), "Page 1 of 2", fontsize=10, color=(0.5, 0.5, 0.5))
    
    # Page 2: Additional content
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 50), "Page 2: More Content", fontsize=20)
    page.insert_text((50, 100), "Additional section with more details", fontsize=12)
    
    content = """This section contains supplementary information.
It includes various content types to test the validation engine.
Different text sizes and styles are used throughout."""
    
    y = 140
    for line in content.split('\n'):
        page.insert_text((50, y), line, fontsize=10)
        y += 20
    
    # Add footer to page 2
    page.insert_text((50, 750), "Page 2 of 2", fontsize=10, color=(0.5, 0.5, 0.5))
    
    doc.save("sample/actual.pdf")
    doc.close()

if __name__ == "__main__":
    print("Creating sample PDFs...")
    create_expected_pdf()
    print("✓ Created sample/expected.pdf")
    create_actual_pdf()
    print("✓ Created sample/actual.pdf")
    print("Sample PDFs created successfully!")
