import os
import shutil
import textwrap
from multiprocessing import cpu_count
from subprocess import Popen, PIPE
from typing import List

from rnaseq_lib3.docker import fix_permissions


def run(df_path: str, group_a: List[str], group_b: List[str], output_dir: str, cores: int = None):
    """
    Runs DESeq2 standard comparison between group A and group B

    Args:
        df_path: Path to a GENES by SAMPLES DataFrame of EXPECTED COUNTS
        group_a: List of samples in group A
        group_b: List of samples in group B
        output_dir: Full path to an output directory
        cores: Number of cores to use. Defaults to # of cores on the machine

    Returns:
        None
    """
    # Check for output to avoid overwriting
    if os.path.exists(os.path.join(output_dir, 'results.tsv')):
        print(f'Output already exists in {output_dir}')
        return None

    # Make workspace directories
    work_dir = os.path.join(output_dir, 'work_dir')
    os.makedirs(work_dir, exist_ok=True)

    # Fix groups based on samples in expected count DataFrame to avoid setcolorder error
    exp_samples = set(open(df_path, 'r').readline().split('\t')[1:])
    group_a = [x for x in group_a if x in exp_samples]
    group_b = [x for x in group_b if x in exp_samples]

    # Write out vectors
    tissue_vector = os.path.join(work_dir, 'tissue.vector')
    with open(tissue_vector, 'w') as f:
        f.write('\n'.join(group_a + group_b))

    disease_vector = os.path.join(work_dir, 'disease.vector')
    with open(disease_vector, 'w') as f:
        f.write('\n'.join(['A' if x in group_a else 'B' for x in group_a + group_b]))

    # Write out script
    cores = cores if cores else int(cpu_count())
    script_path = os.path.join(work_dir, 'deseq2.R')
    with open(script_path, 'w') as f:
        f.write(
            textwrap.dedent("""
            library('DESeq2'); library('data.table'); library('BiocParallel')
            register(MulticoreParam({cores}))
            
            # Argument parsing
            args <- commandArgs(trailingOnly = TRUE)
            df_path <- args[1]
            tissue_path <- args[2]
            disease_path <- args[3]
            output_dir <- '/data/'
            
            # Read in vectors
            tissue_vector <- read.table(tissue_path)$V1
            disease_vector <- read.table(disease_path)$V1
            
            # Read in table and process
            n <- read.table(df_path, sep='\\t', header=1, row.names=1, check.names=FALSE)
            sub <- n[, colnames(n)%in%tissue_vector]
            setcolorder(sub, as.character(tissue_vector))
            
            # Preprocessing
            countData <- round(sub)
            colData <- data.frame(disease=disease_vector, row.names=colnames(countData))
            y <- DESeqDataSetFromMatrix(countData = countData, colData = colData, design = ~ disease)
            
            # Run DESeq2
            y <- DESeq(y, parallel=TRUE)
            res <- results(y, parallel=TRUE)
            summary(res)
            
            # Write out table
            resOrdered <- res[order(res$padj),]
            res_name <- 'results.tsv'
            res_path <- paste(output_dir, res_name, sep='/')
            write.table(as.data.frame(resOrdered), file=res_path, col.names=NA, sep='\\t',  quote=FALSE)
            
            # MA Plot
            ma_name <- 'MA.pdf'
            ma_path <- paste(output_dir, ma_name, sep='/')
            pdf(ma_path, width=7, height=7)
            plotMA(res, main='DESeq2')
            dev.off()
            
            # Dispersion Plot
            disp_name <- 'dispersion.pdf'
            disp_path <- paste(output_dir, disp_name, sep='/')
            pdf(disp_path, width=7, height=7)
            plotDispEsts( y, ylim = c(1e-6, 1e1) )
            dev.off()
            
            # PVal Hist
            hist_name <- 'pval-hist.pdf'
            hist_path <- paste(output_dir, hist_name, sep='/')
            pdf(hist_path, width=7, height=7)
            hist( res$pvalue, breaks=20, col="grey" )
            dev.off()
            
            # Ratios plots
            qs <- c( 0, quantile( res$baseMean[res$baseMean > 0], 0:7/7 ) )
            bins <- cut( res$baseMean, qs )
            levels(bins) <- paste0("~",round(.5*qs[-1] + .5*qs[-length(qs)]))
            ratios <- tapply( res$pvalue, bins, function(p) mean( p < .01, na.rm=TRUE ) )
            ratio_name <- 'ratios.pdf'
            ratio_path <- paste(output_dir, ratio_name, sep='/')
            pdf(ratio_path, width=7, height=7)
            barplot(ratios, xlab="mean normalized count", ylab="ratio of small $p$ values")
            dev.off()                                           
            """.format(cores=cores)))

    # Call DESeq2
    docker_parameters = ['docker', 'run',
                         '-v', '{}:/data'.format(output_dir),
                         '-v', '{}:/df'.format(os.path.dirname(df_path)),
                         'jvivian/deseq2']

    parameters = ['/data/work_dir/deseq2.R',
                  '/df/{}'.format(os.path.basename(df_path)),
                  '/data/{}'.format(os.path.join('work_dir', 'tissue.vector')),
                  '/data/{}'.format(os.path.join('work_dir', 'disease.vector'))]

    print('\nCalling: {}\n'.format(' '.join(docker_parameters + parameters)))
    p = Popen(docker_parameters + parameters, stderr=PIPE, stdout=PIPE, universal_newlines=True)
    out, err = p.communicate()
    if out or err:
        logfile = os.path.join(output_dir, 'log.txt')
        with open(logfile, 'w') as f:
            f.write(f'Number of samples\tA: {len(group_a)}\tB: {len(group_b)}\n\n{out}\n\n{err}')

    # Fix output of files
    fix_permissions(tool='jvivian/deseq2', work_dir=output_dir)

    # Clean up if nothing went wrong
    if p.returncode == 0:
        shutil.rmtree(work_dir)
