#!/usr/bin/env bash
EMAIL_ADDRESS='jtvivian@gmail.com'
COMPENDIUM_VERSION=GTEX
PROTOCOL_VERSION=V13.0.0
docker run --user $UID:`getent group treehouse | cut -d: -f3` --rm -it \
                -v /private/groups/treehouse:/treehouse:ro \
                -v `pwd`/outputs:/work/outputs \
                -v `pwd`:/work/rollup:ro \
                -v `pwd`:/rollup:ro \
                -v `pwd`/manifest.tsv:/app/manifest.tsv:ro \
                protocol:$PROTOCOL_VERSION  \
                    --email $EMAIL_ADDRESS \
                    --inputs /treehouse/archive/downstream \
                    --cohort /treehouse/archive/compendium/$COMPENDIUM_VERSION \
                    --reference /treehouse/archive/references \
                    --outputs /treehouse/archive/projects/pannormal-matching-tissue/outputs
