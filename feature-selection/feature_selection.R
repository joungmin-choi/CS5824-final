library("DESeq2")

data <- read.csv("BRCA_gene_expression_tumor_normal.csv", row.names = 1)
coldata <- read.csv("coldata.csv", row.names = 1)
cts <- as.matrix(data)

coldata$type <- factor(coldata$type)

#Check whether the coldata and cts have same sample names
all(rownames(coldata) %in% colnames(cts))
all(rownames(coldata) == colnames(cts))

dds <- DESeqDataSetFromMatrix(countData = cts, colData = coldata, design = ~ type)
dds <- DESeq(dds)
res <- results(dds)
res <- as.data.frame(res)
deg_df <- res[res$pvalue < 0.01]
deg_df['abs_log2FC'] = abs(deg_df$log2FoldChange)
deg_df <- deg_df[deg_df$abs_log2FC > 1]
write.csv(deg_df, "DEG_list.csv", row.names = T, quote = F)

normalized_counts <- counts(dds, normalized=TRUE)
write.csv(normalized_counts, "normalized_gene_expression.csv", row.names = T, quote = F)