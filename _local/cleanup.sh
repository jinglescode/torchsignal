find . -name '.DS_Store' -type f -delete
find . -name 'Icon?' -type f -delete
find . -name 'package-lock.json' -type f -delete
find . -name "__pycache__" -type d -exec rm -r "{}" \;
