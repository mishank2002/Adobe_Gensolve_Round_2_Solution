import streamlit as st
from svgpathtools import svg2paths2
import numpy as np
import cv2
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from io import BytesIO
import base64
from scipy.optimize import leastsq
import math

# Function to determine the shape of the object
def detect_shape(contour):
    shape = "unidentified"
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    num_vertices = len(approx)
    
    if num_vertices == 3:
        shape = "triangle"
    elif num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif 5 <= num_vertices <= 6:
        shape = "polygon"
    elif num_vertices > 6:
        shape = "polygon"
    else:
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        shape = "circle" if circularity > 0.7 else "ellipse"
    
    return shape

# Function to convert a path to a contour
def path_to_contour(path):
    contour = []
    for seg in path:
        contour.append([int(seg.start.real), int(seg.start.imag)])
        contour.append([int(seg.end.real), int(seg.end.imag)])
    return np.array(contour)

# Function to process SVG and detect shapes
def process_svg(file):
    try:
        file_data = file.read()
        paths, _, _ = svg2paths2(BytesIO(file_data))
        shape_counts = {
            "triangle": 0,
            "square": 0,
            "rectangle": 0,
            "polygon": 0,
            "circle": 0,
            "ellipse": 0,
            "unidentified": 0
        }
        
        for path in paths:
            contour = path_to_contour(path)
            if contour.size == 0:
                continue

            img = np.zeros((1000, 1000), dtype=np.uint8)
            cv2.fillPoly(img, [contour], 255)
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                shape = detect_shape(contour)
                shape_counts[shape] += 1
        
        return shape_counts
    except Exception as e:
        st.error(f"Error processing SVG: {str(e)}")
        return {}

# Function to convert SVG to base64
def svg_to_base64(svg_data):
    encoded = base64.b64encode(svg_data).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"

# Function to fit a circle using least squares
def fit_circle(XY):
    def objective(params):
        x0, y0, r = params
        return np.sqrt((XY[:, 0] - x0) ** 2 + (XY[:, 1] - y0) ** 2) - r

    x_mean, y_mean = np.mean(XY, axis=0)
    r_guess = np.mean(np.sqrt((XY[:, 0] - x_mean) ** 2 + (XY[:, 1] - y_mean) ** 2))
    initial_params = [x_mean, y_mean, r_guess]
    
    result, _ = leastsq(objective, initial_params)
    return result

# Function to detect symmetry in contours
def detect_symmetry(contours):
    def fit_line(points):
        A = np.vstack([points[:, 0], np.ones(len(points))]).T
        m, c = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        return m, c
    
    def compute_symmetry_score(line_params, points):
        m, c = line_params
        distances = np.abs(m * points[:, 0] - points[:, 1] + c) / np.sqrt(m ** 2 + 1)
        return np.mean(distances)

    symmetric_lines = []
    for contour in contours:
        points = np.array(contour).reshape(-1, 2)
        if len(points) < 2:
            continue

        m_horizontal, c_horizontal = fit_line(points)
        score_horizontal = compute_symmetry_score((m_horizontal, c_horizontal), points)
        if score_horizontal < 10:
            symmetric_lines.append(("Horizontal", (m_horizontal, c_horizontal)))

        m_vertical, c_vertical = fit_line(np.flip(points, axis=1))
        score_vertical = compute_symmetry_score((m_vertical, c_vertical), np.flip(points, axis=1))
        if score_vertical < 10:
            symmetric_lines.append(("Vertical", (m_vertical, c_vertical)))

        m_diagonal, c_diagonal = fit_line(points)
        angle = math.degrees(math.atan(m_diagonal))
        if abs(angle - 45) < 5:
            score_diagonal = compute_symmetry_score((m_diagonal, c_diagonal), points)
            if score_diagonal < 10:
                symmetric_lines.append(("Diagonal", (m_diagonal, c_diagonal)))

    return symmetric_lines

# Streamlit pages

def symmetry_detection_page():
    st.header("Symmetry Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symmetric_lines = detect_symmetry(contours)
        st.write(f"Detected {len(symmetric_lines)} lines of symmetry")
        for i, (line_type, (m, c)) in enumerate(symmetric_lines):
            st.write(f"Symmetric Line {i + 1}: {line_type} (y = {m:.2f}x + {c:.2f})")

def occlusion_completion_page():
    st.header("Occlusion Completion")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
        inverted_mask = cv2.bitwise_not(contour_mask)
        completed_image = cv2.bitwise_and(image, image, mask=inverted_mask)

        for contour in contours:
            shape = detect_shape(contour)
            if shape == "circle":
                x0, y0, r = fit_circle(np.array(contour).reshape(-1, 2))
                cv2.circle(completed_image, (int(x0), int(y0)), int(r), (0, 0, 0), -1)
            elif shape == "triangle":
                vertices = contour.reshape(-1, 2)
                cv2.fillPoly(completed_image, [vertices], (0, 0, 0))

        st.image(completed_image, channels="BGR", caption="Completed Occlusion Image")

def regularization_page():
    st.header("Regularization")
    uploaded_file = st.file_uploader("Choose an SVG file", type="svg")

    if uploaded_file is not None:
        try:
            st.write("File uploaded successfully!")
            svg_data = uploaded_file.read()
            base64_svg = svg_to_base64(svg_data)
            st.image(base64_svg, use_column_width=True)
            uploaded_file.seek(0)
            shape_counts = process_svg(uploaded_file)
            st.write(f"Shape Counts: {shape_counts}")
        except Exception as e:
            st.error(f"Failed to process SVG: {str(e)}")

def main():
    st.title("SVG Shape Detection")

    page = st.sidebar.selectbox("Choose a Page", ["Regularization", "Occlusion Completion", "Symmetry Detection"])
    
    if page == "Occlusion Completion":
        occlusion_completion_page()
    elif page == "Regularization":
        regularization_page()
    elif page == "Symmetry Detection":
        symmetry_detection_page()

if __name__ == "__main__":
    main()
