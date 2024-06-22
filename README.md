# Exploring Cybersecurity Data Science

 Cybersecurity incidents have become the norm of the day as there is hardly a day without news of a breach or discovery of illegally obtained data on the dark web. This is despite the
 presence of large amounts of data collected by various devices that are used in the protection of data in information systems. With this high volume of network data that is available
 through several logging systems, human analysts become overwhelmed to manually analyze the data. Even though the data flagged by the monitoring devices is what is thought to
 be malicious, a great portion of this data is comprised of false positives. Most of the time these analysts are cybersecurity professionals with little data analytics/science knowledge.
 If analysts with data analytics/science knowledge are used, they will usually have little cybersecurity knowledge. To aid with finding insights as well as problematic issues from
 the data, there is an increasing use of data mining/data science, machine learning, deep learning, and artificial intelligence methods. The users of these methods should therefore
 be knowledgeable of both data analytics/science and cybersecurity concepts and methods.
 In this paper we will look at how using data analytics techniques can aid in the analysis of cybersecurity data by performing a comparison of dimensionality reduction techniques,
 and of clustering techniques on some cybersecurity datasets


## Methodology
1. NSL_KDD_Data.py and UNSWNB15_Data.py files with classes that preprocess the respective datasets.
2. inputData.py contains a class with functions that create the various input files (raw design matrix. one-hot encoded design matrix, and the gower matrix).
3. DimRed contains a class with functions that create the various dimensionality reduction outputs (PCA, FAMD, UMAP, t-SNE, ISOMAP).
4. Clustering.py contains a class with functions that create the various clustering outputs

   Since the files are very large, each output was saved to a pickle file once available to avoid rerunning. dask parallel processing package was used where methods had that option.
5. nslkddd-01.py and unsw-01.py create the input files, nslkdd_dimRed75.py and unsw_dimRed75.py create the dimemnsionality reduction outputs, cluster_uMap.py create the clustering outputs.

Note: The Anomaly detection section will be added soon.
