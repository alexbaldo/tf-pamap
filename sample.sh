#!/bin/bash
for i in {1..8}
do
	mv data/pamap/processed/subject10$i.dat data/pamap/processed/subject10$i.dat.bak
	l=$(wc -l data/pamap/processed/subject10$i.dat.bak | awk '{print $1;}')
	pct=5
	n=$((l*$pct/100))
	gshuf -n $n data/pamap/processed/subject10$i.dat.bak > data/pamap/processed/subject10$i.dat
done