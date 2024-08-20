/*
  Usage:
  include yp.inc in udf file
  include yp.oudf in the oudf block in udf file
  
  call options (from UDF_ExecuteStep):

      To only get min/max/avg values and use inbuild cheap_dist function for ywd:
      y_p(nrs, time);
      
      To print y_p to file:
      y_p(nrs, time, true)

      User specified wall distance and boundary IDs
      y_p(nrs, time, nbID, o_bID, o_ywd)
      
      or with y_p print to file:
      y_p(nrs, time, nbID, o_bID, o_ywd, true)     
 */

static int outfldCounter = 0;

void y_p(nrs_t *nrs, double time, const bool iffld = false);
void y_p(nrs_t* nrs, double time, const int &nbID, const occa::memory &o_bID, const occa::memory &o_ywd, const bool iffld = false);

dfloat reduce_sum(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat sum = 0.0;
  
  for (dlong i = 0; i < N; i++) {
    sum += tmp[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  return sum;
}

dfloat reduce_min(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat smin = 1e9;
  
  for (dlong i = 0; i < N; i++) {
    smin = fmin(smin, tmp[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &smin, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiComm);

  return smin;
}

dfloat reduce_max(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat smax = -1e9;
  
  for (dlong i = 0; i < N; i++) {
    smax = fmax(smax, tmp[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &smax, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm);

  return smax;
}

void y_p(nrs_t *nrs, double time, const bool iffld)
{
  auto mesh = nrs->meshV;
  
  std::vector<int> wbID;
  occa::memory o_wbID;

  for (auto &[key, bcID] : bcMap::map()) {
    const auto field = key.first;
    if (field == "velocity") {
      if (bcID == bcMap::bcTypeW) {
	wbID.push_back(key.second + 1);
      }
    }
  }
  o_wbID = platform->device.malloc<int>(wbID.size(), wbID.data());
  auto o_ywd = mesh->minDistance(wbID.size(), o_wbID, "cheap_dist");

  y_p(nrs, time, wbID.size(), o_wbID, o_ywd, iffld);
}

void y_p(nrs_t* nrs, double time, const int &nbID, const occa::memory &o_bID, const occa::memory &o_ywd, const bool iffld)
{
  auto mesh = nrs->meshV;
  
  auto o_Sij = nrs->strainRate();
  
  auto o_miny = platform->o_memPool.reserve<dfloat>(mesh->Nelements);
  auto o_maxy = platform->o_memPool.reserve<dfloat>(mesh->Nelements);
  auto o_sumy = platform->o_memPool.reserve<dfloat>(mesh->Nelements);
  
  occa::memory o_yp = o_NULL;
  if(iffld) o_yp = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset); 
  
  yplusKernel(mesh->Nelements,
	      nrs->fieldOffset,
	      nbID,
	      o_bID,
	      static_cast<int>(iffld),
	      mesh->o_sgeo,
	      mesh->o_vmapM,
	      mesh->o_EToB,
	      nrs->o_rho,
	      nrs->o_mue,
	      o_Sij,
	      o_ywd,
	      o_yp,
	      o_miny,
	      o_maxy,
	      o_sumy);

  static std::vector<dfloat> tmp;
  if (tmp.size() < o_sumy.size()) {
    tmp.resize(o_sumy.size());
  }

  auto o_tmp2 = platform->o_memPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(mesh->Nlocal, 1.0, o_tmp2);
  const auto areaWall = mesh->surfaceAreaMultiplyIntegrate(nbID, o_bID, o_tmp2);

  o_sumy.copyTo(tmp.data());
  const dfloat sumy = reduce_sum(tmp, mesh->Nelements);

  o_miny.copyTo(tmp.data());
  const dfloat miny = reduce_min(tmp, mesh->Nelements);

  o_maxy.copyTo(tmp.data());
  const dfloat maxy = reduce_max(tmp, mesh->Nelements);

  if(platform->comm.mpiRank == 0)
    std::cout << std::scientific << "\n      y+ min: " << miny <<" max: "<< maxy <<" avg: "<< sumy/areaWall.at(0)<<"\n";

  if(iffld){
    int outXYZ = 0;
    if(!outfldCounter) outXYZ = 1;
    std::vector<occa::memory> o_out;
    o_out.push_back(o_yp);
    fld::write("yplus", time, outfldCounter, o_out, outXYZ, 1);
    outfldCounter++;
  }
}
