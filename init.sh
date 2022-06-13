pip install -r requirements.txt

CURDIR=$(cd "$(dirname "$0")"; pwd) 
ln -s ${CURDIR}/data/ ${CURDIR}/src/data
