suppressPackageStartupMessages(library(fst));suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE);root<-normalizePath('.');source<-readLines('data/metadata/latest_tracerx_public.txt',warn=FALSE);p<-file.path(source,'processed')
stamp<-format(Sys.time(),'%Y%m%dT%H%M%S',tz='UTC');out<-file.path('data/metadata',paste0('tracerx_native_',stamp));dir.create(out)
write_tsv<-function(x,name){con<-gzfile(file.path(out,paste0(name,'.tsv.gz')),'wt');write.table(x,con,sep='\t',row.names=FALSE,quote=FALSE,na='NA');close(con)}
mut<-read_fst(file.path(p,'20221109_TRACERx421_mutation_table.fst'));reg<-read_fst(file.path(p,'20221123_TRACERx421_mutation_table_region.fst'))
pat<-readRDS(file.path(p,'20221109_TRACERx421_all_patient_df.rds'));tum<-readRDS(file.path(p,'20221109_TRACERx421_all_tumour_df.rds'));trees<-readRDS(file.path(p,'20221109_TRACERx421_phylogenetic_trees.rds'))
qc<-list(patients=nrow(pat),tumours=nrow(tum),mutations=nrow(mut),region_mutation_rows=nrow(reg),trees=length(trees),mutation_patient_unmatched=sum(!mut$patient_id %in% pat$cruk_id),mutation_tumour_unmatched=sum(!mut$tumour_id %in% tum$tumour_id_muttable_cruk),region_mutation_id_unmatched=sum(!reg$mutation_id %in% mut$mutation_id),tree_tumour_unmatched=sum(!names(trees) %in% tum$tumour_id_muttable_cruk),mutations_duplicate_ids=sum(duplicated(mut$mutation_id)),region_count=length(unique(reg$RegionID)))
fields<-list();for (name in c('mut','reg','pat','tum')){d<-get(name);fields[[name]]<-data.frame(file=name,field=names(d),rows=nrow(d),missing=vapply(d,function(x)sum(is.na(x)),numeric(1)))}
write_tsv(do.call(rbind,fields),'field_missingness');write_tsv(mut,'mutations');write_tsv(reg,'region_mutations');write_tsv(pat,'patients');write_tsv(tum,'tumours')
edges<-list();tree_objects<-list();mutation_cluster<-list();tree_qc<-list()
for (id in names(trees)){
 v<-trees[[id]];g<-v$graph_pyclone;entry<-list(tumour_id=v$tumour_id,trunk=g$trunk,corrected_edges=unname(g$Corrected_tree),original_edges=unname(g$Original_tree),alternative_trees=lapply(g$alt_trees,unname),consensus_relationships=g$consensus_relationships,consensus_branches=g$consensus_branches,tree_removed_clusters=v$tree_removed_clusters,copy_number_removed_clusters=v$cpn_removed_clusters,merged_clusters=v$merged_clusters)
 tree_objects[[id]]<-entry
 for (kind in c('Corrected_tree','Original_tree')){e<-g[[kind]];if(!is.null(e)&&length(e)>0){edges[[length(edges)+1]]<-data.frame(tumour_id=id,kind=kind,parent=e[,1],child=e[,2])}}
 for (k in seq_along(g$alt_trees)){e<-g$alt_trees[[k]];if(length(e)>0)edges[[length(edges)+1]]<-data.frame(tumour_id=id,kind=paste0('alternative_',k),parent=e[,1],child=e[,2])}
 cc<-v$ccf_table_pyclone_clean;mutation_cluster[[id]]<-data.frame(tumour_id=id,variant_key=rownames(cc),cluster=cc$PycloneCluster)
 tree_qc[[id]]<-data.frame(tumour_id=id,trunk=paste(g$trunk,collapse=';'),corrected_edges=if(is.null(g$Corrected_tree))0 else nrow(g$Corrected_tree),alternate_trees=length(g$alt_trees),mutation_cluster_rows=nrow(cc))
}
write_tsv(do.call(rbind,edges),'tree_edges');write_tsv(do.call(rbind,mutation_cluster),'tree_mutation_clusters');write_tsv(do.call(rbind,tree_qc),'tree_inventory')
write_json(tree_objects,file.path(out,'tree_objects.json'),auto_unbox=TRUE,pretty=FALSE,na='null',null='null',matrix='rowmajor')
write_json(qc,file.path(out,'summary.json'),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'));writeLines(out,'data/metadata/latest_tracerx_native.txt')
cat(toJSON(qc,auto_unbox=TRUE,pretty=TRUE),'\nOUTPUT',out,'\n')
