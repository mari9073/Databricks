# Databricks notebook source
from pyspark.sql.functions import col, when, to_date, lit
from pyspark.sql.types import StringType,BooleanType,DateType
import pandas as pd

# COMMAND ----------

cdb_i_ct_rlps = spark.sql('SELECT * FROM hive_metastore.default.cdb_i_ct_rlps_2')  # 계약관계자정보
cdb_i_ct_cont_pvt = spark.sql('SELECT * FROM hive_metastore.default.cdb_i_ct_cont_pvt_2')  # 보험계약정보

# COMMAND ----------

# MAGIC %md
# MAGIC # 계약관계자정보(cdb_i_ct_rlps)

# COMMAND ----------

cdb_i_ct_rlps.display()

# COMMAND ----------

cdb_i_ct_rlps.count()

# COMMAND ----------

print('join_sn : ', cdb_i_ct_rlps.select('join_sn').distinct().count())
print('join_sn_typ : ', cdb_i_ct_rlps.select('join_sn_typ').distinct().count())
print('COM_SN : ', cdb_i_ct_rlps.select('COM_SN').distinct().count())
print('POL_SN : ', cdb_i_ct_rlps.select('POL_SN').distinct().count())
print('SCTR_CD : ', cdb_i_ct_rlps.select('SCTR_CD').distinct().count())

# COMMAND ----------

import pandas as pd
cdb_i_ct_rlps.toPandas().JOIN_SN.value_counts()

# COMMAND ----------

cdb_i_ct_rlps.filter(cdb_i_ct_rlps.JOIN_SN == 1554239).display()

# COMMAND ----------

cdb_i_ct_rlps.toPandas().COM_SN.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # 보험계약정보(cdb_i_ct_cont_pvt)

# COMMAND ----------

cdb_i_ct_cont_pvt.display()

# COMMAND ----------

print(cdb_i_ct_cont_pvt.select('SCTR_CD').distinct().count())
print(cdb_i_ct_cont_pvt.select('POL_SN').distinct().count())

# COMMAND ----------

cdb_i_ct_cont_pvt.select('pol_sn').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Join (계약관계자정보+보험계약정보)

# COMMAND ----------

print(cdb_i_ct_rlps.count())
print(cdb_i_ct_rlps.filter(cdb_i_ct_rlps.CT_RLPS_GBN_CD=='C').count())  #계약자
print(cdb_i_ct_rlps.filter(cdb_i_ct_rlps.CT_RLPS_GBN_CD=='I').count())   #피보험자

# COMMAND ----------

rlps_df = cdb_i_ct_rlps.filter(cdb_i_ct_rlps.CT_RLPS_GBN_CD=='I')

# COMMAND ----------

rlps_df.display()

# COMMAND ----------

print(rlps_df.select('JOIN_SN').distinct().count())
print(rlps_df.select('POL_SN').distinct().count())

# COMMAND ----------

rlps_df.toPandas().POL_SN.value_counts()

# COMMAND ----------

rlps_df.filter(rlps_df.POL_SN=='750334').display()

# COMMAND ----------

join_df = rlps_df.join(cdb_i_ct_cont_pvt, on=['SCTR_CD', 'POL_SN'], how='inner')
join_df.display()

# COMMAND ----------

print(rlps_df.count())
print(cdb_i_ct_cont_pvt.count())
print(join_df.count())

# COMMAND ----------

rlps_df.filter(rlps_df.POL_SN=='8054287').display()

# COMMAND ----------

cdb_i_ct_cont_pvt.filter(cdb_i_ct_cont_pvt.POL_SN=='8054287').display()

# COMMAND ----------

join_df.filter(join_df.POL_SN=='8054287').display()

# COMMAND ----------

join_df.filter(join_df.POL_SN=='8054287').count()

# COMMAND ----------

print(join_df.filter(join_df.CT_TRMNT_DT=='999912').count())  # 계약해지 x
print(join_df.filter(join_df.CT_TRMNT_DT!='999912').count())  # 계약해지 o

# COMMAND ----------

join_df.toPandas().CT_PY_CYCLE_CD.value_counts()

# COMMAND ----------

df = join_df.select('SCTR_CD', 'POL_SN', 'JOIN_SN', 'COM_SN', 'IS_ME', 'CT_IN_RLTN_CD', 'is_indiv', 'GIS_CD', 'IS_GRP_INS', 'INS_GBN_CD', 'INS_CT_STAT_CD', 'CT_CNCLS_DT', 'CT_TRMNT_DT', 'CT_END_DT', 'CT_EFCTV_DT', 'CT_RST_DT', 'CT_PY_AMT', 'CT_PY_CYCLE_CD', 'CT_PY_PD', 'IS_CT_DIGNS', 'CT_JDGMT_CD', 'CT_CHNL_CD')

# COMMAND ----------

df.dtypes

# COMMAND ----------

df2 = df.withColumn("POL_SN", col("POL_SN").cast(StringType()))
df2 = df2.withColumn("JOIN_SN", col("JOIN_SN").cast(StringType()))
df2 = df2.withColumn("COM_SN", col("COM_SN").cast(StringType()))
df2 = df2.withColumn("CT_CNCLS_DT", to_date(col("CT_CNCLS_DT"),"yyyyMM"))
df2 = df2.withColumn("CT_END_DT", to_date(col("CT_END_DT"),"yyyyMM"))

# COMMAND ----------

df2.display()

# COMMAND ----------

# 해지 - N / 유지 - Y
df3 = df2.select(col('*'), when(df2.CT_TRMNT_DT == '999912', 'Y').otherwise('N').alias('CT_TRMNT_DT_Target'))

# COMMAND ----------

df3 = df3.select(col('*'), when(df3.CT_EFCTV_DT != '999912', '888812').otherwise(df3.CT_EFCTV_DT).alias('CT_EFCTV_DT_2'))
df3 = df3.select(col('*'), when(df3.CT_RST_DT != '999912', '888812').otherwise(df3.CT_RST_DT).alias('CT_RST_DT_2'))
df3 = df3.select(col('*'), when(df3.CT_END_DT == '9999-12-01', '2222-12-01').otherwise(df3.CT_END_DT).alias('CT_END_DT_2'))

# COMMAND ----------

df3 = df3.withColumn("CT_END_DT_2", to_date(col("CT_END_DT_2"),"yyyy-MM-dd"))

# COMMAND ----------

df3.display()

# COMMAND ----------

df4 = df3.drop('CT_TRMNT_DT', 'CT_EFCTV_DT', 'CT_RST_DT', 'CT_END_DT')
df4.display()

# COMMAND ----------

df4.count()

# COMMAND ----------

train, test = df4.randomSplit([0.8, 0.2])

# COMMAND ----------

print(df4.count())
print(train.count())
print(test.count())

# COMMAND ----------

train.dtypes

# COMMAND ----------

# train.write.saveAsTable("CDB_I_CT_train")
# test.write.saveAsTable("CDB_I_CT_test")

# COMMAND ----------

train.display()

# COMMAND ----------

train.write.mode("overwrite").saveAsTable("CDB_I_CT_train")
test.write.mode("overwrite").saveAsTable("CDB_I_CT_test")

# COMMAND ----------

# %sql
# DROP TABLE IF EXISTS default.cdb_i_ct_rlps_1

# COMMAND ----------

