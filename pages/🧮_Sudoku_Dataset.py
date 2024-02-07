import pandas as pd
import numpy as np
import streamlit as st

def main():
    st.title("Sudoku Dataset")

    # Load the first few rows of the dataset
    data = pd.read_csv('sudoku.csv', nrows=100)

    # Select a puzzle
    puzzle_index = st.selectbox("Select a puzzle", data.index)
    puzzle = data.loc[puzzle_index, 'quizzes']
    solution = data.loc[puzzle_index, 'solutions']

    # Convert the puzzle string and solution string to 2D arrays
    grid_puzzle = np.array([int(digit) for digit in puzzle]).reshape((9, 9))
    grid_solution = np.array([int(digit) for digit in solution]).reshape((9, 9))

    # Convert the 2D arrays to pandas DataFrames
    df_puzzle = pd.DataFrame(grid_puzzle)
    df_solution = pd.DataFrame(grid_solution)

    # Display the puzzle and solution
    st.markdown("<h2 style='text-align: center;'>Puzzle</h2>", unsafe_allow_html=True)
    st.table(df_puzzle)
    st.markdown("<h2 style='text-align: center;'>Solution</h2>", unsafe_allow_html=True)
    st.table(df_solution)

if __name__ == "__main__":
    main()