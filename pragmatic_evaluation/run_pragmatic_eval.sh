#!/bin/sh
filenames=`ls ../../acl_results/predictions/three_features/*.csv`
for entry in $filenames
do
  echo "$entry"
  save_dir=$( echo "$entry" |cut -d'/' -f6 )
  echo $save_dir
  python pragmatic_eval.py -rf="$entry" -o="../../acl_results/pragmatics/false_features_test/three_features/$save_dir"
  #python img2text_inference.py -pr="similar" -ri -ft="$entry" -idf="data/3dshapes_test_IDs.pt"
done