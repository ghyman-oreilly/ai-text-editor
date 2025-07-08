
# AI Editor Script

A script for using AI to edit documents in alignment with an editorial stylesheet.

## Limitations

* Only Asciidoc file format (.asciidoc or .adoc) currently supported.

## TODO

* Some post-processing of responses:
  * Check for backtickets (```r'^[`]*|[`]*$'```) enclosing response, where not present in original
  * Check for language tag where not in the original
  * URL format? Caption format? Headings? We might be able to regex this stuff instead of doing a QA pass 

  
