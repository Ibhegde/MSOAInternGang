rm -rf vit-result
rm -rf vit-result.zip

cp -r $1 vit-result
rm -rf vit-result/*/runs
rm -rf vit-result/*/checkpoint-*
echo $1 > vit-result/release.txt
zip -r vit-result.zip vit-result/

