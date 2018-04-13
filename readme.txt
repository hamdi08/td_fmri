Dependency (for the .m files):
 MATLAB Tensor Toolbox(we experimented with Version 2.6) 
 Download the zipped file from http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html
 Extract the folder and add the folder's path in MATLAB

Data can be downloaded from https://goo.gl/7jnAS3
Data for ADHD (used in tensor_subject_rois_ts_adhd.m) -- subjects_768_adhd.mat 
 -- In experiment, 48 samples (indexed by 85:131) are used only

Data for reading disability -- DS2.mat (sensitive medical data; not available in the abovementioned shared folder; contact author if you need it anyway)

acc_all.mat -- accuracies of SVM classification after 7-fold cross validation on the subject factor matrix found after CP and Tucker decomposition on reading disability data
acc_all_adhd.mat -- accuracies of SVM classification after 8-fold cross validation on the subject factor matrix found after CP and Tucker decomposition on ADHD data
