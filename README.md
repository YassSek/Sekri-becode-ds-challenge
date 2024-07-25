# Web Scraping Tool

## Overview

A simple and efficient web scraping tool designed to extract data from RTBF : https://www.rtbf.be/en-continu
periodically to store articles content links and title

Using camemBERT model for analysing article values then creating cluster and visualisation to see different subject
....

## Features

- **Scraping news from rtbf en continu**
- **stored title , link and content in csv**
- **preprocessing title and tokenisation using camemBERT models**
- **Clustering with Kmeans**
- **Visualisation with Matplotlib**

## Installation

Clone the repository and install the required dependencies ( in requierements.txt ) , run the scrap script first and then run the main.py

if you clone this repository on github you could use sheduled scraping ( modify the yml file to change time of recurrence )
