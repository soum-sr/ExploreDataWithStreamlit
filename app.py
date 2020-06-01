import streamlit as st 
import os
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

def select_dataset_file():
	dataset_dir = './datasets'
	filenames = os.listdir(dataset_dir)
	selected_dataset = st.selectbox("",filenames)
	return os.path.join(dataset_dir, selected_dataset)

def check_dataset_category(filename):
	f_name = filename[11:]
	regression = ['BostonHousing.csv']
	sequence = ['livestock.csv','guinearice.csv']
	classification = ['wine.csv','iris.csv', 'Zoo.csv','abalone.csv','BreastCancer.csv']
	if f_name in regression:
		return 'Regression'
	if f_name in sequence:
		return 'Sequence'
	if f_name in classification:
		return 'Classification'

def main():
	# Title 
	st.title('Explore Standard Machine Learning Datasets')
	# Dropdown option to select the dataset
	st.write("### Select a Dataset from below: ")
	selected_dataset = select_dataset_file()
	dataset_type = check_dataset_category(selected_dataset)
	st.info(f'Dataset Type: {dataset_type}')

	# Read the csvfile
	df = pd.read_csv(selected_dataset)

	# display the dataset
	if st.checkbox("Show Dataset"):
		st.write("### Enter the number of rows to view")
		rows = st.number_input("", min_value=0,value=0)
		if rows > 0:
			st.dataframe(df.head(rows))

	# Option to check the column names
	if st.checkbox("Show dataset with selected columns"):
		columns = df.columns.tolist()
		st.write("### Select the columns to display")
		selected_cols = st.multiselect("", columns)
		if len(selected_cols) > 0:
			selected_df = df[selected_cols]
			st.dataframe(selected_df)

	# Show the dimension of the dataframe
	if st.checkbox("Show number of rows and columns"):
		st.write(f'Rows: {df.shape[0]}')
		st.write(f'Columns: {df.shape[1]}')

	# description of dataset
	if st.checkbox("Show description of dataset"):
		st.write(df.describe())

	# Value count of target values
	if dataset_type == 'Classification':
		if st.checkbox("Show Pie Chart and Value Counts of Target Columns"):
			st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()
			st.write(df.iloc[:,-1].value_counts())

	# Coorelation plot of dataset
	if dataset_type == 'Regression' or dataset_type == 'Classification':
		if st.checkbox("Show Coorelation Plot"):
			st.write("### Heatmap")
			fig, ax = plt.subplots(figsize=(10,10))
			st.write(sns.heatmap(df.corr(), annot=True,linewidths=0.5))
			st.pyplot()

	# Sequence plot for sequential dataset
	if dataset_type == "Sequence":
		columns = df.columns.tolist()
		df.set_index(columns[0], inplace=True)
		if st.checkbox("Plot Sequence Data"):
			plot_type = st.selectbox("Select type of plot: ", ["area", "line", "bar"])
			if st.button("Generate"):
				if plot_type == "area":
					st.area_chart(df)
				if plot_type == "line":
					st.line_chart(df)
				if plot_type == "bar":
					st.bar_chart(df)



	# Select plotting style: area, bar, line, hist, box

if __name__ == '__main__':
	main()