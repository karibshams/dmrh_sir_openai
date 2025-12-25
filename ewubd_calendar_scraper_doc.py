# ewubd_calendar_scraper_doc.py

import requests
from bs4 import BeautifulSoup
from docx import Document

URL = "https://ewubd.edu/academic-calendar-details/spring-undergraduate-2025"
OUTPUT_FILE = "Spring_2025_Undergraduate_Academic_Calendar.docx"


def scrape_calendar(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    data = []
    rows = table.find_all("tr")[1:]  # skip header

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) == 3:
            data.append(cols)

    return data


def save_to_doc(data, filename):
    doc = Document()
    doc.add_heading("Academic Calendar – Spring 2025 (Undergraduate)", level=1)

    table = doc.add_table(rows=1, cols=3)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Date"
    hdr_cells[1].text = "Day"
    hdr_cells[2].text = "Event"

    for row in data:
        row_cells = table.add_row().cells
        row_cells[0].text = row[0]
        row_cells[1].text = row[1]
        row_cells[2].text = row[2]

    doc.save(filename)


if __name__ == "__main__":
    calendar_data = scrape_calendar(URL)
    save_to_doc(calendar_data, OUTPUT_FILE)
    print(f"Saved {len(calendar_data)} rows to {OUTPUT_FILE}")
