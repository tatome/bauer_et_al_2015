SHELL=/bin/bash
PDFLATEX=pdflatex --shell-escape --extra-mem-bot=10000000
LATEXMK=latexmk -pdf -latex='${PDFLATEX} %O %S'
PYTHON=/usr/bin/python2
GREP=/bin/grep
SED=/bin/sed
export PYTHONPATH=attention/code:attention/code/evaluation

INCLUDEDTEX= \
	preamble.tex \
	symbols.tex macros.tex drafting.tex \
	tikzstuff.tex \
	abbreviations.tex \
	attention.tex introduction.tex network.tex experiment.tex conclusion.tex

MAINDOC=attention
DELETABLES=fdb_latexmk fls log aux auxlock dvi bbl blg toc
BIBFILE=bib.bib

NORMALSIM=attention/simulations/normal
NORMALEV=attention/evaluations/normal

GENERATEDTEX=attention_simulation_data.tex diptable.tex diptable-sens.tex diptable-gains.tex

IMAGES=

GRAPHS=graphs/enhancement_spatial.pgf graphs/enhancement_feature.pgf \
		graphs/spatial_activation.pgf graphs/mapping.pgf graphs/integration_by_distance.pgf graphs/enhancement_cognitive_by_sensory.pgf

TIKZPICTURES=tikzpictures/network-attention.tex tikzpictures/enhancement_cognitive_by_sensory.tex \
		tikzpictures/enhancement_feature.tex tikzpictures/integration_by_distance.tex

COMPILEDGRAPHS=graphs/enhancement_spatial0.pdf graphs/spatial_activation0.pdf graphs/mapping0.pdf
		

COMPILEDTIKZPICTURES=tikzpictures/network-attention0.pdf tikzpictures/enhancement_cognitive_by_sensory0.pdf \
		tikzpictures/enhancement_feature0.pdf tikzpictures/integration_by_distance0.pdf

.PHONY: all
all: ${MAINDOC}.pdf

${MAINDOC}.pdf: ${MAINDOC}.tex ${GENERATEDTEX} ${COMPILEDGRAPHS} ${COMPILEDTIKZPICTURES} ${INCLUDEDTEX} ${BIBFILE}
	${LATEXMK} ${MAINDOC}.tex

SIMULATIONDATA= %/feature.csv.bz2 %/incongruent.csv.bz2 %/mapping.csv.bz2 %/network.pickle.bz2 \
				%/simulation.log %/spatial.csv.bz2 %/incongruent-spatial.csv.bz2
${SIMULATIONDATA}: %/simulation_done
	@echo "Simulation data in $(@D) up to date."
.PRECIOUS: ${SIMULATIONDATA}

SIMULATION_CODE=$(wildcard attention/code/*.py)

%/simulation_done: ${SIMULATION_CODE} %/config.yaml
	@echo "Running simulation in $(@D).  This will take a long time.  Hit <CTRL> + C to abort."
	sleep 10
	${PYTHON} attention/code/attention.py \
		-o $(@D) \
		-l $(@D)/simulation.log \
		--processes -1
	touch $@
.PRECIOUS: %/simulation_done

attention/evaluations/%/mapping.npy: attention/code/evaluation/compute_mapping.py attention/simulations/%/mapping.csv.bz2
	${PYTHON} $< -c $(dir $(word 2,$^))/config.yaml -i $(word 2,$^) -o $@
.PRECIOUS: attention/evaluations/%/mapping.npy

attention_simulation_data.tex: attention/code/evaluation/simulationdata.py ${NORMALSIM}/config.yaml ${NORMALEV}/spatial_attention_effect.yaml
	${PYTHON} $< -s ${NORMALSIM}/config.yaml -a ${NORMALEV}/spatial_attention_effect.yaml -e ${NORMALEV}/config.yaml -o $@

attention/evaluations/normal/spatial_attention_effect.yaml: \
		attention/code/evaluation/compute_spatial_attention_effect.py \
		${NORMALSIM}/config.yaml \
		${NORMALEV}/mapping.npy \
		${NORMALSIM}/incongruent-spatial.csv.bz2
	${PYTHON} $< -c ${NORMALSIM}/config.yaml \
					-m ${NORMALEV}/mapping.npy \
					-i ${NORMALSIM}/incongruent-spatial.csv.bz2 \
					-o $@

diptable.tex: attention/code/evaluation/make_integration_by_distance_table.py \
			$(addsuffix /integration_statistics.npz,$(wildcard attention/evaluations/noise/*)) \
			attention/evaluations/normal/integration_statistics.npz
	${PYTHON} $< -i $(wordlist 2,1000,$^) -o $@ 

diptable-sens.tex: attention/code/evaluation/make_integration_by_distance_table_sens.py \
			$(addsuffix /integration_statistics.npz,$(wildcard attention/evaluations/sensory/*)) \
			attention/evaluations/normal/integration_statistics.npz
	${PYTHON} $< -i $(wordlist 2,1000,$^) -o $@ 

diptable-gains.tex: attention/code/evaluation/make_integration_by_distance_table_gains.py \
			$(addsuffix /integration_statistics.npz,$(wildcard attention/evaluations/gains/*)) \
			attention/evaluations/normal/integration_statistics.npz
	${PYTHON} $< -i $(wordlist 2,1000,$^) -o $@ 

attention/evaluations/%/incongruent_data.npz: \
		attention/code/evaluation/convert_to_npy.py \
		attention/simulations/%/incongruent.csv.bz2 \
		attention/simulations/%/config.yaml \
		attention/evaluations/%/mapping.npy
	${PYTHON} $< -i $(word 2,$^) -c $(word 3,$^) -m $(word 4,$^) -o $@
.PRECIOUS: attention/evaluations/%/incongruent_data.npz

attention/evaluations/%/integration_statistics.npz: \
		attention/code/evaluation/compute_integration_statistics.py \
		attention/evaluations/%/incongruent_data.npz
	${PYTHON} $< -i $(word 2,$^) -o $@

graphs/integration_by_distance.pgf: attention/code/evaluation/plot_integration_by_distance.py \
									${NORMALEV}/incongruent_data.npz
	rm -f tikzpictures/integration_by_distance0.pdf
	${PYTHON} $< \
			-i $(word 2,$^) \
			-o $@ \
			--figsize 4.5 5.5

graphs/enhancement_spatial.pgf: attention/code/evaluation/plot_spatial_enhancement.py ${NORMALSIM}/config.yaml ${NORMALEV}/mapping.npy ${NORMALSIM}/spatial.csv.bz2
	rm -f graphs/enhancement_spatial0.pdf
	${PYTHON} attention/code/evaluation/plot_spatial_enhancement.py -c ${NORMALSIM}/config.yaml -e ${NORMALEV}/config.yaml -m ${NORMALEV}/mapping.npy -i ${NORMALSIM}/spatial.csv.bz2  --figsize 5 1.5 -s $@
 
${NORMALEV}/enhancement_numerical.yaml: attention/code/evaluation/evaluate_enhancement.py ${NORMALSIM}/config.yaml ${NORMALEV}/config.yaml ${NORMALEV}/mapping.npy ${NORMALSIM}/feature.csv.bz2
	${PYTHON} attention/code/evaluation/evaluate_enhancement.py \
			-c ${NORMALSIM}/config.yaml \
			-e ${NORMALEV}/config.yaml \
			-m ${NORMALEV}/mapping.npy \
			-i ${NORMALSIM}/feature.csv.bz2 \
			-o $@

graphs/enhancement_feature.pgf: attention/code/evaluation/plot_feature_enhancement.py ${NORMALSIM}/config.yaml ${NORMALEV}/config.yaml ${NORMALEV}/enhancement_numerical.yaml
	rm -f `basename $@ .pgf`0.pdf
	${PYTHON} attention/code/evaluation/plot_feature_enhancement.py \
			-c ${NORMALSIM}/config.yaml \
			-d ${NORMALEV}/enhancement_numerical.yaml \
			-e ${NORMALEV}/config.yaml \
			--figsize 4.5 2.5 \
			-o $@

graphs/enhancement_cognitive_by_sensory.pgf: attention/code/evaluation/plot_sensory_to_feature_enhancement.py ${NORMALSIM}/config.yaml ${NORMALEV}/config.yaml ${NORMALEV}/enhancement_numerical.yaml
	rm -f `basename $@ .pgf`0.pdf
	${PYTHON} attention/code/evaluation/plot_sensory_to_feature_enhancement.py \
			-c ${NORMALSIM}/config.yaml \
			-d ${NORMALEV}/enhancement_numerical.yaml \
			-e ${NORMALEV}/config.yaml \
			--figsize 4.5 5.2 \
			-o $@

graphs/enhancement_sensory.pgf: attention/code/evaluation/plot_sensory_enhancement.py ${NORMALEV}/enhancement_numerical.yaml
	rm -f graphs/enhancement_sensory0.pdf
	${PYTHON} attention/code/evaluation/plot_sensory_enhancement.py -i ${NORMALEV}/enhancement_numerical.yaml --figsize 5 1.8 -o $@

graphs/enhancement_feature_to_sensory.pgf: attention/code/evaluation/plot_enhancement_feature_to_sensory_enhancement.py ${NORMALEV}/enhancement_numerical.yaml
	rm -f `basename $@ .pgf`0.pdf
	${PYTHON} attention/code/evaluation/plot_enhancement_feature_to_sensory_enhancement.py -i ${NORMALEV}/enhancement_numerical.yaml --figsize 5 1.8 -o $@

graphs/spatial_activation.pgf: attention/code/evaluation/plot_spatial_activation.py attention/code/activation_functions.py ${NORMALSIM}/config.yaml
	rm -f graphs/spatial_activation0.pdf
	${PYTHON} attention/code/evaluation/plot_spatial_activation.py -c ${NORMALSIM}/config.yaml --figsize 5 1.5 -o $@

graphs/mapping.pgf: attention/code/evaluation/plot_mapping.py ${NORMALEV}/mapping.npy
	rm -f graphs/mapping0.pdf
	${PYTHON} attention/code/evaluation/plot_mapping.py --figsize 5 1.5 -m ${NORMALEV}/mapping.npy -o $@

# Compile graphs.
${COMPILEDGRAPHS}: ${GRAPHS}
	rm -f $@
	${PDFLATEX} ${MAINDOC}.tex

tikzpictures/network-attention0.pdf: tikzpictures/network-attention.tex
	rm -f $@
	${PDFLATEX} ${MAINDOC}.tex

tikzpictures/enhancement_cognitive_by_sensory0.pdf: tikzpictures/enhancement_cognitive_by_sensory.tex graphs/enhancement_cognitive_by_sensory.pgf
	rm -f $@
	${PDFLATEX} ${MAINDOC}.tex

tikzpictures/enhancement_feature0.pdf: tikzpictures/enhancement_feature.tex graphs/enhancement_feature.pgf
	rm -f $@
	${PDFLATEX} ${MAINDOC}.tex

tikzpictures/integration_by_distance0.pdf: tikzpictures/integration_by_distance.tex graphs/integration_by_distance.pgf
	rm -f $@
	${PDFLATEX} ${MAINDOC}.tex

bauer-magg-wermter-2015.pdf bauer-magg-wermter-2015.tar.bz2: ${MAINDOC}.pdf
	rm -f bauer-magg-wermter-2015.pdf bauer-magg-wermter-2015.tar.bz2
	rm -f submittable/*
	${PYTHON} scripts/make_submittable.py -m attention.tex -o submittable/bauer-magg-wermter-2015.tex
	tar cjf bauer-magg-wermter-2015.tar.bz2 submittable 
	(cd submittable && ${LATEXMK} bauer-magg-wermter-2015.tex )
	mv submittable/bauer-magg-wermter-2015.pdf .

bauer-magg-wermter-2015-all-code.tar.bz2:
	( \
		rm -f $@ && \
		mkdir TEMPDIR && \
		cd TEMPDIR && \
		git clone .. code && \
		find code -name simulation_done | xargs rm && \
		rm -rf code/.git && \
		tar cjf ../$@ code && \
		cd .. && \
		rm -rf TEMPDIR \
	)

.PHONY: clean
clean:
	rm -f graphs/*
	rm -f *.log *.pdf *.aux *.bbl *.auxlock *.blg *.fbd_latexmk
	(cd tikzpictures; rm -f *.log *.pdf *.aux *.bbl *.auxlock *.blg *.fbd_latexmk *.dpth)
	rm -f ${GENERATEDTEX}

