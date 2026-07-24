library(doParallel)
library(BuenColors)
library(FigR)
library(FNN) # FigR 依赖这个包
library(BSgenome.Hsapiens.UCSC.hg19)
library(SummarizedExperiment)
library(GenomicRanges)
library(Seurat)
library(Signac)
library(cowplot)
library(data.table)
library(ggplot2)
library(patchwork)
library(stringr)
library(Matrix)


cell_type <- "Panc1"

data_path <- paste0("/home/liyang/BioWuYan/dygmamba_project/data/cell_line/", cell_type, "/process/")

output_path <- paste0("/home/liyang/BioWuYan/dygmamba_project/data/cell_line/", cell_type, "/data_FigR/")

dir.create(output_path, recursive = TRUE, showWarnings = FALSE)

atac_dir <- paste0(data_path,  "R/atac/")

rna_dir <- paste0(data_path,  "R/rna/")

############################################################################################
#******************************** Data Read --- ATAC ***************************************
############################################################################################

mtx <- readMM(paste0(atac_dir, "sparse.mtx"))

cellinfo <- fread(paste0(atac_dir, "cellinfo.csv"), header = T, data.table = F)

atacinfo <- fread(paste0(atac_dir, "atacinfo.csv"), header = T, data.table = F)

rownames(mtx) <- cellinfo$V1
colnames(mtx) <- atacinfo$V1

meta <- cellinfo
rownames(meta) <- cellinfo$V1

if(identical(rownames(meta), rownames(mtx))){
        # 行需为 feature
    print("Info identical, Can create Seurat Object")
    mtx_transposed <- t(mtx) 

    atac_seurat <- CreateSeuratObject(count = mtx_transposed, meta.data = meta, assay = "ATAC")
}

############################################################################################
#******************************** Data Read --- RNA ***************************************
############################################################################################


rna_mtx <- readMM(paste0(rna_dir, "sparse.mtx"))

rna_cellinfo <- fread(paste0(rna_dir, "cellinfo.csv"), header = T, data.table = F)

rna_info <- fread(paste0(rna_dir, "rnainfo.csv"), header = T, data.table = F)

rownames(rna_mtx) <- rna_cellinfo$V1
colnames(rna_mtx) <- rna_info$Geneid

meta <- rna_cellinfo
rownames(meta) <- rna_cellinfo$V1


    
print("Create Seurat Object")
rna_mtx_transposed <- t(rna_mtx)

rna_seurat <- CreateSeuratObject(count = rna_mtx_transposed, meta.data = meta, assay = "RNA")

############################################################################################
#******************************** Data process ***************************************
############################################################################################

atac_seurat <- RunTFIDF(atac_seurat)
atac_seurat <- FindTopFeatures(atac_seurat, min.cutoff = 'q0')
atac_seurat <- RunSVD(atac_seurat)

cell_embeddings <- Embeddings(atac_seurat, reduction = "lsi")[, 2:30]
k_value <- 30
knn_result <- get.knn(data = cell_embeddings, k = k_value)
cellKNN.mat <- knn_result$nn.index

rownames(cellKNN.mat) <- rownames(cell_embeddings)

# 假设您的 Peak 名称格式为 "chr1-100-200" 或 "chr1:100-200"
peak_names <- rownames(atac_seurat)
peak_ranges <- StringToGRanges(peak_names, sep = c("-", "-")) # 根据您的实际分隔符调整

ATAC.SE <- SummarizedExperiment(
  assays = list(counts = GetAssayData(atac_seurat, assay = "ATAC", layer = "counts")),
  rowRanges = peak_ranges
)
rowData(ATAC.SE) <- atac_seurat[["ATAC"]][[]]

###############################################

if (DefaultAssay(rna_seurat) != "RNA") DefaultAssay(rna_seurat) <- "RNA"
rna_seurat <- NormalizeData(rna_seurat)

rnaMat <- GetAssayData(rna_seurat, assay = "RNA", layer = "data") 

common_cells <- intersect(rownames(cellKNN.mat), colnames(rnaMat))

if(length(common_cells) == 0) {
  stop("错误：ATAC 和 RNA 数据没有共同的细胞名！请检查两个 Seurat 对象的细胞名格式。")
}

cellKNN.mat.sub <- cellKNN.mat[common_cells, ]
rnaMat.sub <- rnaMat[, common_cells]

check_result <- all.equal(rownames(cellKNN.mat.sub), colnames(rnaMat.sub))
print(paste("细胞对齐检查:", check_result))

rnaMat.smooth <- smoothScoresNN(NNmat = cellKNN.mat.sub, 
                                mat = rnaMat.sub, 
                                nCores = 4)

cat("RNA 平滑完成！维度:", dim(rnaMat.smooth), "\n")


############################################################################################
#******************************** DOROC ***************************************
############################################################################################

cisCor <- runGenePeakcorr(ATAC.se = ATAC.SE,
                          RNAmat = rnaMat,
                          genome = "hg38", 
                          nCores = 4)

# 筛选显著的关联
cisCor.filt <- cisCor %>% filter(pvalZ <= 0.05)

# 确定 DORC 基因列表 (默认相关峰数阈值为 7)
dorcGenes <- cisCor.filt %>% dorcJPlot(cutoff = 2, returnGeneList = TRUE)

# 4. 计算并平滑 DORC 得分
# 提示：cellKNN.mat 是细胞间的近邻矩阵，通常可通过 Seurat 的 FindNeighbors 结果构建
dorcMat <- getDORCScores(ATAC.SE, dorcTab = cisCor.filt, geneList = dorcGenes, nCores = 4)
dorcMat.smooth <- smoothScoresNN(NNmat = cellKNN.mat, mat = dorcMat, nCores = 4)

############################################################################################
#******************************** DOROC ***************************************
############################################################################################
# 运行 FigR
# rnaMat.smooth 建议使用经过平滑处理的 RNA 矩阵以提高信噪比
fig.d <- runFigRGRN(ATAC.se = ATAC.SE,
                    rnaMat = rnaMat.smooth, 
                    dorcMat = dorcMat.smooth,
                    dorcTab = cisCor.filt,
                    genome = "hg38",
                    dorcGenes = dorcGenes,
                    nCores = 4)


############################################################################################
#******************************** Result analysis ***************************************
############################################################################################

region_gene_df <- cisCor.filt[, c("PeakRanges", "Gene", "rObs", "pvalZ")]

colnames(region_gene_df) <- c("Region", "Target_Gene", "Correlation", "P_Value")

write.csv(region_gene_df, paste0(output_path, "Region_Gene_Network.csv"), row.names = FALSE)

tf_gene_df <- fig.d %>% 
  filter(abs(Score) >= 1.0) %>%
  select(Motif, DORC, Score, Corr, Enrichment.P)

colnames(tf_gene_df) <- c("TF", "Target_Gene", "Regulation_Score", "Correlation", "Enrichment_PVal")

write.csv(tf_gene_df, paste0(output_path, "TF_Gene_Network.csv"), row.names = FALSE)