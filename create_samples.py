"""Create sample expected.pdf and actual.pdf for testing."""
import fitz  # PyMuPDF

def create_expected_pdf():
    """Create the expected PDF with baseline content."""
    doc = fitz.open()
    
    # Page 1: Title and content
    page = doc.new_page(width=612, height=792)
    
    # Header
    page.insert_text((50, 50), "PDF Validation Test Report", fontsize=24, color=(0, 0, 0))
    page.insert_text((50, 80), "Expected Document", fontsize=12, color=(100, 100, 100))
    
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
    img_rect = fitz.Rect(50, 320, 200, 420)
    page.draw_rect(img_rect, color=(0, 0, 255), fill=(100, 150, 255))
    page.insert_text((50, 440), "Figure 1: Test Image", fontsize=10)
    
    # Insert a table
    rows = [
        ["Column A", "Column B", "Column C"],
        ["Data 1.1", "Data 1.2", "Data 1.3"],
        ["Data 2.1", "Data 2.2", "Data 2.3"],
        ["Data 3.1", "Data 3.2", "Data 3.3"],
    ]
    
    table_rect = fitz.Rect(50, 480, 500, 600)
    x, y = 50, 480
    row_height = 30
    col_width = 150
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_rect = fitz.Rect(x + j * col_width, y + i * row_height, 
                                 x + (j + 1) * col_width, y + (i + 1) * row_height)
            page.draw_rect(cell_rect, color=(0, 0, 0))
            page.insert_text((cell_rect.x0 + 5, cell_rect.y0 + 10), cell, fontsize=9)
    
    page.insert_text((50, 630), "Table 1: Sample Data", fontsize=10)
    
    # Page 2: More content
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((50, 50), "Page 2 Content", fontsize=14)
    page2.insert_text((50, 80), "Additional information for validation testing.", fontsize=10)
    
    doc.save("sample/expected.pdf")
    doc.close()
    print("✓ Created sample/expected.pdf")

def create_actual_pdf():
    """Create the actual PDF with intentional differences."""
    doc = fitz.open()
    
    # Page 1: With differences
    page = doc.new_page(width=612, height=792)
    
    # Header (slightly different text)
    page.insert_text((50, 50), "PDF Validation Test Report", fontsize=24, color=(0, 0, 0))
    page.insert_text((50, 80), "Actual Document (Modified)", fontsize=12, color=(100, 100, 100))
    
    # Content section with changes
    page.insert_text((50, 120), "Introduction", fontsize=14)
    text = """This is a modified test document for PDF validation.
It contains different content types for comparison.
The validation engine checks for differences in:
• Text content and position
• Images and their sizes
• Tables and cell data
• Overall visual rendering
• Additional line of text not in original"""
    y = 140
    for line in text.split('\n'):
        page.insert_text((50, y), line, fontsize=10)
        y += 15
    
    # Insert image at different position with different size
    img_rect = fitz.Rect(60, 330, 210, 420)  # Different position and size
    page.draw_rect(img_rect, color=(0, 200, 0), fill=(150, 255, 150))  # Different color
    page.insert_text((50, 440), "Figure 1: Modified Image", fontsize=10)  # Different label
    
    # Insert table with changes
    rows = [
        ["Column A", "Column B", "Column C"],
        ["Data 1.1", "Changed 1.2", "Data 1.3"],  # Changed cell
        ["Data 2.1", "Data 2.2", "Data 2.3"],
        ["Data 3.1", "Data 3.2", "Modified 3.3"],  # Changed cell
    ]
    
    table_rect = fitz.Rect(50, 480, 500, 600)
    x, y = 50, 480
    row_height = 30
    col_width = 150
    
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_rect = fitz.Rect(x + j * col_width, y + i * row_height, 
                                 x + (j + 1) * col_width, y + (i + 1) * row_height)
            page.draw_rect(cell_rect, color=(0, 0, 0))
            page.insert_text((cell_rect.x0 + 5, cell_rect.y0 + 10), cell, fontsize=9)
    
    page.insert_text((50, 630), "Table 1: Sample Data (Updated)", fontsize=10)
    
    # Page 2: Different structure
    page2 = doc.new_page(width=612, height=792)
    page2.insert_text((50, 50), "Page 2 Content - Modified", fontsize=14)
    page2.insert_text((50, 80), "This page has been significantly altered.", fontsize=10)
    page2.insert_text((50, 110), "Extra content added.", fontsize=10)
    
    doc.save("sample/actual.pdf")
    doc.close()
    print("✓ Created sample/actual.pdf")

if __name__ == "__main__":
    create_expected_pdf()
    create_actual_pdf()
    print("\nSample PDFs created successfully!")
