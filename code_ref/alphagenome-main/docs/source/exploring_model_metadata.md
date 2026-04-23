# Model output metadata

AlphaGenome returns predictions for 11 different output types, covering a
variety of modalities. Here we provide details about the human model outputs and
associated metadata to help users make informed decisions about the parameters
of their API requests (e.g., ontology term; output types).

For further details on dataset processing and precise definitions of each output
type, including their respective units and normalization methods, please refer
to the Methods section of the AlphaGenome paper.

<!-- mdformat off(Turn off mdformat to retain grid card syntax.) -->
<!-- mdlint off(LINK_ID) -->

{#model_metadata-table1-target}
**Table 1**: Descriptions of output types predicted by AlphaGenome.

| OutputType name | Description | Units | Resolution | Unique biosamples | Total tracks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [RNA\_SEQ](#alphagenome.models.dna_output.OutputType.RNA_SEQ) | RNA expression as measured by RNA-seq. Includes a mixture of PolyA+ RNA and Total RNA assays. Some tracks are also stranded. | Normalized read signal | 1bp | 285 | 667 |
| [CAGE](#alphagenome.models.dna_output.OutputType.CAGE) | RNA expression at transcription start-sites as measured by Cap Analysis Gene Expression (CAGE) assay. | Normalized read signal | 1bp | 264 | 546 |
| [PROCAP](#alphagenome.models.dna_output.OutputType.PROCAP) | RNA expression at transcription start-sites as measured by Precision Run-On sequencing and capping (PROCAP) assay. | Normalized read signal | 1bp | 6 | 12 |
| [DNASE](#alphagenome.models.dna_output.OutputType.DNASE) | Chromatin accessibility as measured by DNase I hypersensitive sites sequencing (DNase-seq) assay. | Normalized insertion signal | 1bp | 305 | 305 |
| [ATAC](#alphagenome.models.dna_output.OutputType.ATAC) | Chromatin accessibility as measured by the transposase-accessible chromatin (ATAC-seq) assay. | Normalized insertion signal | 1bp | 167 | 167 |
| [CHIP\_HISTONE](#alphagenome.models.dna_output.OutputType.CHIP_HISTONE) | Relative abundance of histone modification marks as measured by chromatin immunoprecipitation (ChIP-seq) for 24 different markers e.g. H3k27ac (see ENCODE [documentation](https://www.encodeproject.org/chip-seq/histone/)). | Fold-change over control (summed per 128 bp bin) | 128bp | 219 | 1116 |
| [CHIP\_TF](#alphagenome.models.dna_output.OutputType.CHIP_TF) | Relative abundance of DNA-bound transcription factors as measured by ChIP-seq targeting 43 different proteins (see ENCODE [documentation](https://www.encodeproject.org/chip-seq/transcription_factor/)). | Fold-change over control (summed per 128 bp bin) | 128bp | 163 | 1617 |
| [SPLICE\_SITES](#alphagenome.models.dna_output.OutputType.SPLICE_SITES) | Predicted location of donor or acceptor splice sites, for both the positive and negative strand, expressed as a probability (higher numbers indicate higher probability of the base being a splice site). | Predicted probability | 1bp | NA | 4 |
| [SPLICE\_JUNCTIONS](#alphagenome.models.dna_output.OutputType.SPLICE_JUNCTIONS) | Splice junction spliced read counts, as measured by RNA-Seq. Predictions are for all possible pairings of at most 512 donors and 512 acceptors from each strand in the requested interval, where the position of donors and acceptors along the input sequence is given by predictions of splice site positions. | Normalized junction signal | 1bp | 282 | 734 |
| [SPLICE\_SITE\_USAGE](#alphagenome.models.dna_output.OutputType.SPLICE_SITE_USAGE) | Fraction of transcripts using a splice site, as measured by RNA-seq. All reads that span a given splice site are considered, and we predict the fraction of these that use the site (donor or acceptor). | Fraction | 1bp | 282 | 734 |
| [CONTACT\_MAPS](#alphagenome.models.dna_output.OutputType.CONTACT_MAPS) | Relative frequency of physical contact between pairwise positions (symmetric), derived from chromatin contact maps (Micro-C and Hi-C assays). Values are coarse-grained and normalized by removing the off-diagonal power law decay (as also done in [Zhou, J. 2022](https://www.nature.com/articles/s41588-022-01065-4)). | Log-fold over genomic distance-based expectation | 2048bp | 12 | 28 |
<!-- mdlint on -->
<!-- mdformat on -->

## Track metadata

To access the metadata describing each track for human outputs use:

```py
output_metadata = dna_model.output_metadata(
    organism=dna_client.Organism.HOMO_SAPIENS
)
```

Each predicted output type (e.g., RNA\_SEQ) contains metadata in a
{py:class}`~pandas.DataFrame`: {py:class}`output_metadata.rna_seq
<alphagenome.models.dna_output.OutputMetadata.rna_seq>`

Each row of the {class}`~pandas.DataFrame` corresponds to a ‘track’, and each
column contains key information for biological interpretation such as:

*   `name`: Name of the track. Example: `CL:0000047 polyA plus RNA-seq`.
*   `strand` Strand of the track, either positive (`+`), negative (`-`), or
    unstranded (`.`).
*   `ontology_curie`: A string ID representing the ontology term corresponding
    to the biosample. Example: `CL:0000100`.
*   `biosample_name`: Plain text description of the biosample. Example: `motor
    neuron`.

For a full list of metadata columns available for each output type, please see
the [navigating data ontologies notebook](colabs/tissue_ontology_mapping), which
demonstrates how to access and browse track metadata.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

:::{note} For `SPLICE_JUNCTION` outputs the strand information is a property of
a junction rather than a track, so the metadata for this output type will show
half as many rows as reported in the above table.
:::

<!-- mdformat on -->

## Additional track metadata

Some output types contain additional columns. For example,
{py:class}`OutputMetadata.rna_seq
<alphagenome.models.dna_output.OutputMetadata.rna_seq>` and
{py:class}`OutputMetadata.splice_sites
<alphagenome.models.dna_output.OutputMetadata.splice_sites>` also contain a
`gtex_tissue` column, which is populated for the tracks that make predictions
for the tissues sampled in the
[GTEx project](https://gtexportal.org/home/samplingSitePage)
{cite:t}`gtex2020gtex`.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->
:::{note} For one tissue,
’Brain \- Cerebellar hemisphere’, we used an alternative Uberon ID to that was
provided in the
[GTEx documentation](https://gtexportal.org/home/samplingSitePage)
(‘UBERON:0002037’), to reflect Uberon’s ID for cerebellar hemisphere:
[‘UBERON:0002245'](https://www.ebi.ac.uk/ols4/ontologies/uberon/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FUBERON_0002245).
:::

<!-- mdformat on -->
