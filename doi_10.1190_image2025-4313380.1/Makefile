# Makefile for building the review findings PDF from Quarto source.
#
# Prerequisites:
#   - Quarto (https://quarto.org)
#   - TeX Live with LuaLaTeX or pdfLaTeX
#
# Usage:
#   make          # Regenerate figures then build PDF
#   make pdf      # Build PDF only (assumes figures exist)
#   make figures  # Regenerate all figures from Python scripts
#   make clean    # Remove build artifacts

QUARTO   := quarto
PYTHON   := python3
QMD      := review_findings.qmd
PDF      := review_findings.pdf

# Python stage scripts (ordered by dependency)
STAGES := stage0_operator_analysis.py \
          stage1_acoustic_dual_pair.py \
          stage2_elastic_isotropic.py \
          stage3_filtering_and_source.py \
          stage4_tti_elastic.py

# Generated figure files (representative subset for dependency tracking)
FIGS := figures/fig2_dispersion_dissipation.png \
        figures/fig3_filter_response.png \
        figures/stage1_convergence.png \
        figures/stage2_two_layer_elastic.png \
        figures/stage2_energy_conservation.png \
        figures/stage3_fig1_reproduction.png \
        figures/stage3_fig1_6panel.png \
        figures/stage4_anisotropy_comparison.png

# Paper figure extractions
PAPER_FIGS := figures/paper/page2_img1.png \
              figures/paper/page2_img2.png \
              figures/paper/page3_img1.png

.PHONY: all pdf figures clean

# Default target: regenerate figures then build PDF
all: figures pdf

pdf: $(PDF)

# Build PDF from Quarto
$(PDF): $(QMD) $(FIGS) $(PAPER_FIGS)
	$(QUARTO) render $(QMD) --to pdf

# Regenerate all figures by running each stage script
figures: $(FIGS)

figures/fig2_dispersion_dissipation.png figures/fig3_filter_response.png: stage0_operator_analysis.py
	$(PYTHON) $<

figures/stage1_convergence.png figures/stage1_pulse_comparison.png figures/stage1_two_layer.png: stage1_acoustic_dual_pair.py
	$(PYTHON) $<

figures/stage2_two_layer_elastic.png figures/stage2_energy_conservation.png figures/stage2_explosion_source.png: stage2_elastic_isotropic.py
	$(PYTHON) $<

figures/stage3_fig1_reproduction.png figures/stage3_filter_comparison.png figures/stage3_fig1_6panel.png: stage3_filtering_and_source.py
	$(PYTHON) $<

figures/stage4_anisotropy_comparison.png figures/stage4_vti.png figures/stage4_tti_tilted.png figures/stage4_isotropic_limit.png: stage4_tti_elastic.py
	$(PYTHON) $<

# Extract paper figures from the original PDF
figures/paper/page2_img1.png figures/paper/page2_img2.png figures/paper/page3_img1.png: | figures/paper
	$(PYTHON) -c "\
import fitz, os; \
doc = fitz.open('$(HOME)/Downloads/image2025-4313380.1.pdf'); \
os.makedirs('figures/paper', exist_ok=True); \
[open(os.path.join('figures/paper', f'page{p+1}_img{i+1}.{doc.extract_image(img[0])[\"ext\"]}'), 'wb').write(doc.extract_image(img[0])['image']) for p in range(len(doc)) for i, img in enumerate(doc[p].get_images(full=True))]; \
doc.close()"

figures/paper:
	mkdir -p figures/paper

clean:
	rm -f $(PDF)
	rm -rf review_findings_files/
	rm -f *.log *.aux *.out *.toc
