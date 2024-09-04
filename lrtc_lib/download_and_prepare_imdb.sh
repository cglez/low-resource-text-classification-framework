#!/bin/bash
set -e

if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
fi

curr_dir=$(dirname "$0")
base_dir=$(realpath "${curr_dir}"/data)
raw_dir=$(realpath "${base_dir}"/raw)
avail_dir=$(realpath "$base_dir"/available_datasets)
python_path=$(realpath "${curr_dir}"/..)
force_new="" && [ "$1" = "--force-new" ] && force_new="$1"

if [ ! -d "$raw_dir" ]; then
    mkdir "$raw_dir"
fi
cd "$raw_dir"
echo "$raw_dir"

##### imdb
if [ ! -f imdb/test.tsv ] ; then
  echo '** Downloading IMDB files **'
  mkdir -p imdb
  imdb_base_url="https://github.com/forest-snow/alps/raw/main/data/imdb/"
  curl -LJO "$imdb_base_url"/train.tsv
  curl -LJO "$imdb_base_url"/dev.tsv
  curl -LJO "$imdb_base_url"/test.tsv
  mv train.tsv dev.tsv test.tsv imdb/
fi

if [ -f imdb/test.tsv ] && [ ! -f ../available_datasets/imdb/test.csv ] ; then
  mkdir -p "$avail_dir"/imdb
  python "$base_dir"/prepare_imdb.py
fi

cd "$python_path"
python -m lrtc_lib.data.load_dataset $force_new imdb
