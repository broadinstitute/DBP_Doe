#echo "Hello World"
mkdir -p model/model1
python FromScratch.py --training_source sourcedata --training_target targetdata --model_path model --number_of_epochs 200 --testing_source 18_C1_all.tif --testing_target 18_C1_all_masks.tif --output_directory output --source_path 070721_Slide2_Animal1_all.tif
#python /test.py --training_source 17_C1_all.tiff --training_target 17_C1_all_masks.tif  --model_path model --number_of_epochs 1 --testing_source 17_C1_all.tiff --testing_target 17_C1_all_masks.tif --output_directory output --source_path 17_C1_all.tiff

tar -czf output.tar.gz output
tar -czf model.tar.gz model
