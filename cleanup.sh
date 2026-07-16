#!/bin/bash

DIR="/home/zoltas/person_detect/files/buffer"
DIR_ANNOT="/home/zoltas/person_detect/files/buff_annot"
DIR_DAMAGED="/home/zoltas/person_detect/files/damaged"
MAX_FILES=1000

FILE_COUNT=$(find "$DIR" -type f | wc -l)

if [ "$FILE_COUNT" -gt "$MAX_FILES" ]; then

    FILES_TO_DELETE=$((FILE_COUNT - MAX_FILES))

    echo "Usuwam $FILES_TO_DELETE najstarszych plików..."

    find "$DIR" -type f -printf '%T@ %p\n' \
        | sort -n \
        | head -n "$FILES_TO_DELETE" \
        | cut -d' ' -f2- \
        | while read file; do
            rm -f "$file"
          done
          
      find "$DIR_ANNOT" -type f -printf '%T@ %p\n' \
        | sort -n \
        | head -n "$FILES_TO_DELETE" \
        | cut -d' ' -f2- \
        | while read file; do
            rm -f "$file"
          done

      find "$DIR_DAMAGED" -type f -printf '%T@ %p\n' \
        | sort -n \
        | head -n "$FILES_TO_DELETE" \
        | cut -d' ' -f2- \
        | while read file; do
            rm -f "$file"
          done

fi
