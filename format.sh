if [ $# -eq 0 ]
then
    echo 'usage: ./format.sh [python files]'
fi

for i in $@
do
    echo $i
    yapf --style=format.ini $i > tmp_${i%.py}
    mv tmp_${i%.py} $i
done
