#!/usr/bin/env bash
#
#  Specify a URL for a picture and this script will fetch and classify it.
#

set -e

url=$1
if [ -z $url ]; then
    echo "Syntax: $(basename $0) https://www.something.com/image.jpg"
    exit 1
fi

ext=$(rev <<< "$url" | cut -d. -f 1 | rev)

input_name="input.$ext"

curl -o "$input_name" "$url"

sudo ./imagenet.py "$input_name" -v

rm "$input_name"
