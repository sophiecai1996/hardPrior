# Priors, population sizes, and power in genome-wide hypothesis tests

Genome-wide tests always carry a high multiple testing burden. This burden can be overcome by enrolling larger cohorts, or by using prior biological knowledge to favor some hypotheses over others. However, there ’s no quantitative comparison between these 2 methods, especially when the rapid advance of sequencing technology has enabled a much larger cohort size.

Here we compared these 2 methods. We found that increasing the population size is exponentially more valuable than increasing the strength of prior. Analysis of GWAS data shows a 1.4-fold annual increase of sample size, which will outperform a well developed prior-based method with the power to limit testing to 1% of the original tests pool. Given that the sophisticated analysis itself requires 1-2 years effort, prior-based method can hardly keep up with the natural growing of cohort size.


This repository contains the data analysis for GWAS catalog.

## Dataset

Data used for analysis are contained in data_catalog/.

**chr_length_grch38p13.txt**: Human reference genome chromosome length and accession number, gained from NCBI GenBank

**gwas_catalog_v1.0.2-associations_e100_r2020-06-13.tsv**: Datasets were collected from the GWAS Catalog (accessed on June 13, 2020).

**trait_abbreviation.txt**: Selected traits and corresponding abbreviations. Traits are selected from results/disease_pmid_summary.txt with the following steps:

- Studies were grouped into phenotypes based on the trait vocabulary.
- For every phenotype, we then arranged the studies chronologically by publication date and retained studies reporting more findings than all previous studies. 
- Only phenotypes with at least 3 effective studies were kept for further analysis.

Columns in the file:

- Trait: trait name.
- Abbreviation: trait abbreviation.
- dx, dy: manually added. Controlling the jitter for fig_doublingtime.png, only used for plots.
- n_tot: incooperated from results/disease_pmid_summary.txt. Total number of studies related to the particular trait in GWAS catalog.
- n_informative: incooperated from results/disease_pmid_summary.txt. Number of effective studies for thee particular trait.
- recent_pmid, recent_year, recent_size, recent_nsignif:  incooperated from results/disease_pmid_summary.txt. The pubmed id, published year, cohort size and number of significant findings for the most recent study of the trait.
- size_tau, size_se, size_pval: incooperated from results/disease_mooreslaw.txt. Linear regression statistics for cohort size. Doubling time, standard variation of cohort size, p-value for linear regression of cohort size against time.
- nsignif_tau, nsignif_se, nsignif_pva: incooperated from results/disease_mooreslaw.txt. Linear regression statistics for numbers of significant findings. Doubling time, standard variation of number of SNPs, p-value for linear regression of number of SNPs against time (The significant findings reported by GWAS catalog has already accounted for linkage disequilibrium).
- taudiff_z, taudiff_p, taudiff_str: incooperated from results/disease_mooreslaw.txt. Hypothesis testing for doubling time of cohort size and number of findings. Z-score, p-value, and description of the difference (cohort size > loci: discovering; cohort size = loci: tie; cohort size < loci: saturation).

Note: when the abbreviation file is firstly used, only the trait and abbreviation is needed. All the other information is incorporated for convenience.

## Execution

Analyze GWAS dataset and get cohort size and loci doubling time.

```bash
python make_gwasfigures.py --catalog data_catalog/gwas_catalog_v1.0.2-associations_e100_r2020-06-13.tsv --traitabbrev_file data_catalog/trait_abbreviation.txt --chrfile data_catalog/chr_length_grch38p13.txt --outdir ./results
```

Theoratical analysis for relationship between effect of prior strength and cohort size.

```bash
python prior_vs_population.py --pval 5e-8 --power 0.8 --outdir ./analytical
```

## Output

GWAS analysis:

- disease_mooreslaw.txt: linear regression of cohort size/number of loci against time.
- disease_pmid_summary.txt: summary of studies for each traits recorded in GWAS catalog.
- fig_doublingtime.pdf: comparison between doubling time of cohort size and number of loci.
- model_compare/: regression model comparison. Linear model vs null model. Quadratic model vs linear model.
- plot_manhattan/: manhattan plots for each selected trat.
- plot_moore/: linear regression plots of cohort size and number of loci for each selected trait.

