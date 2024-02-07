# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sudoku_solver import batch_smart_solve
from image_recognition import read_cells, preprocess, main_outline, reframe, splitcells, CropCell
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Sudoku solver model
solver_model_path = "sudoku_solver_model.h5"
solver_model = load_model(solver_model_path)

# Load the image recognition model using the new function
recognition_model_path = "image_recognition_model.h5"
recognition_model = load_model(recognition_model_path)

# Streamlit app
def main():
    st.title("Sudoku Solver App")

    # Use st.cache to cache the preprocess function
    @st.cache_resource
    def preprocess_cached(_image):
        image_array = np.array(image)
        sudoku_a = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        threshold = preprocess(sudoku_a)

        contour, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        biggest, _ = main_outline(contour)
        if biggest.size != 0:
            biggest = reframe(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imagewrap = cv2.warpPerspective(sudoku_a, matrix, (450, 450))
            imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

        return imagewrap, threshold, contour

    # Streamlit app
    st.title("Sudoku Solver App")

    # Upload image
    uploaded_file = st.file_uploader("Choose a Sudoku image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Image recognition
        st.header("Image Recognition")
        st.write("Processing...")

        # Preprocess the image for recognition using the cached function
        imagewrap, threshold, contour = preprocess_cached(_image=image)

        # Split cells
        sudoku_cell = splitcells(imagewrap)
        sudoku_cell_croped = CropCell(sudoku_cell)

        # Read cells using the image recognition model
        grid = read_cells(sudoku_cell_croped, recognition_model)

        # Reshape the grid to a 9x9 matrix
        grid = np.reshape(grid, (9, 9))

        # Display the recognized Sudoku grid
        st.subheader("Recognized Sudoku Grid")
        st.write(grid)

        # Sudoku Solver
        st.header("Sudoku Solver")
        st.write("Solving...")

        # Confidence threshold for the solver model
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        # Use st.cache to cache the solver function
        @st.cache(allow_output_mutation=True)
        def batch_smart_solve_cached(_solver, grid, confidence_threshold):
            return batch_smart_solve(_solver, grid, confidence_threshold)

        # Solve the Sudoku grid using the solver model with caching and specified confidence threshold
        solved_grid = batch_smart_solve_cached(np.expand_dims(grid, axis=0), solver_model, confidence_threshold=confidence_threshold)

        # Display the solved Sudoku grid
        st.subheader("Solved Sudoku Grid")
        st.write(solved_grid[0])

        # Offer another solution option
        if st.button("Get Another Solution"):
            solved_grid = batch_smart_solve_cached(np.expand_dims(grid, axis=0), solver_model, confidence_threshold=confidence_threshold)
            st.subheader("Another Solution")
            st.write(solved_grid[0])

        # Include dataset analysis visualization in the sidebar
        st.sidebar.subheader("Dataset Analysis")
        if st.sidebar.checkbox("Visualize Dataset Analysis"):
            if uploaded_file is not None:
                visualize_analysis(imagewrap, threshold, contour, sudoku_cell_croped)

# ...
def visualize_analysis(imagewrap, threshold, contour, sudoku_cell_croped):
    st.header("Dataset Analysis Visualization")
    
    st.write("Step pertama")
    fig, ax = plt.subplots()
    ax.imshow(threshold, cmap='gray')
    st.pyplot(fig)

    st.write("Step kedua")
    fig, ax = plt.subplots()
    contour_image = np.zeros_like(imagewrap)
    cv2.drawContours(contour_image, contour, -1, (255, 255, 255), 1)
    ax.imshow(contour_image, cmap='gray')
    st.pyplot(fig)

    st.write("Step ketiga")
    fig, ax = plt.subplots()
    ax.imshow(imagewrap, cmap='gray')
    st.pyplot(fig)

    st.write("Crop Cells")
    st.write("There are {} crop cells:".format(len(sudoku_cell_croped)))
    
    # Create a 9x9 grid of subplots
    fig, axes = plt.subplots(9, 9, figsize=(12, 12))
    for i in range(9):
        for j in range(9):
            ax = axes[i, j]
            cell_index = i * 9 + j
            if cell_index < len(sudoku_cell_croped):
                ax.imshow(sudoku_cell_croped[cell_index], cmap='gray')
                ax.axis('off')
                ax.set_title("Cell {}".format(cell_index+1))
            else:
                ax.axis('off')  # Hide empty subplots
            
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    main()