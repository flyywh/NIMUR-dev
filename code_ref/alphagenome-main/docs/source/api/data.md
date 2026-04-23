# Data

Classes and utilities for manipulating genomics data.

## Fold Intervals

``` {eval-rst}
.. currentmodule:: alphagenome
```

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.fold_intervals.Subset
    data.fold_intervals.get_all_folds
    data.fold_intervals.get_fold_names
    data.fold_intervals.get_fold_intervals
```

## Genome

``` {eval-rst}
.. currentmodule:: alphagenome
```

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.genome.Strand
    data.genome.Interval
    data.genome.Variant
    data.genome.Junction
```

## Gene annotation

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.gene_annotation.TranscriptType
    data.gene_annotation.extract_tss
    data.gene_annotation.filter_transcript_type
    data.gene_annotation.filter_protein_coding
    data.gene_annotation.filter_to_longest_transcript
    data.gene_annotation.filter_to_mane_select_transcript
    data.gene_annotation.filter_transcript_support_level
    data.gene_annotation.get_gene_interval
    data.gene_annotation.get_gene_intervals
```

## Junction data

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.junction_data.JunctionData
    data.junction_data.get_junctions_to_plot
```

## Ontology

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.ontology.OntologyType
    data.ontology.OntologyTerm
```

## Track data

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.track_data.TrackData
    data.track_data.concat
    data.track_data.interleave
    data.track_data.metadata_to_proto
    data.track_data.metadata_from_proto
    data.track_data.from_protos
```

## Transcript

``` {eval-rst}

.. autosummary::
    :toctree: generated

    data.transcript.Transcript
    data.transcript.TranscriptExtractor
```
