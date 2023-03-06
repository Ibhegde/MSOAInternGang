TRAIN_IMAGES=$1

mv $TRAIN_IMAGES/activity_type/validation/Billboard/Transit/* $TRAIN_IMAGES/activity_type/validation/Billboard/
mv $TRAIN_IMAGES/activity_type/train/Billboard/Transit/* $TRAIN_IMAGES/activity_type/train/Billboard/
mv $TRAIN_IMAGES/activity_type/test/Billboard/Transit/* $TRAIN_IMAGES/activity_type/test/Billboard/
rename 's/Billboard/Billboard-Transit/' $TRAIN_IMAGES/activity_type/*/*

mv $TRAIN_IMAGES/activity_type/validation/Magazine/Newspaper/* $TRAIN_IMAGES/activity_type/validation/Magazine/
mv $TRAIN_IMAGES/activity_type/train/Magazine/Newspaper/* $TRAIN_IMAGES/activity_type/train/Magazine/
mv $TRAIN_IMAGES/activity_type/test/Magazine/Newspaper/* $TRAIN_IMAGES/activity_type/test/Magazine/
rename 's/Magazine/Magazine-Newspaper/' $TRAIN_IMAGES/activity_type/*/*

