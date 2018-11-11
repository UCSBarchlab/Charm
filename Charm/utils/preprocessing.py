from pandas import *

class IOHelper(object):

    @staticmethod
    def read_data(path=None):
        """Read data from given path.
           See usage example for file format.
        """
        if not path:
            raise ValueError('Path to empirical data file not specified!')
        data = read_csv(path, header=2)
        workloads = list(data.groupby('workload').grouper.groups.keys())
        return data, workloads

    @staticmethod
    def read_data_from_cpudb(path=None):
        """Read data from cpu db.
        """

        # fix path
        if not path:
            raise ValueError('Path to CPUDB not specified!')
        if path[-1] != '/':
            path += '/'

        # read data
        processor_data = read_csv(path + 'processor.csv', quotechar = '"')
        specint_1992 = read_csv(path + 'spec_int1992.csv', quotechar = '"')
        specint_1995 = read_csv(path + 'spec_int1995.csv', quotechar = '"')
        specint_2000 = read_csv(path + 'spec_int2000.csv', quotechar = '"')
        specint_2006 = read_csv(path + 'spec_int2006.csv', quotechar = '"')

        # merge dataframes
        specint_data = specint_2006.merge(specint_2000, how='outer', on='processor_id', suffixes=('_06', '')).merge(specint_1995, how='outer', on='processor_id', suffixes=('_00', '')).merge(specint_1992, how='outer', on='processor_id', suffixes=('_95', '_92'))

        merged_data = processor_data.merge(specint_data, how='outer', left_on='id', right_on='processor_id')
        #print merged_data.columns.values

        # determine scaling factors
        spec92to95 = np.mean(merged_data['basemean_95'][merged_data.basemean_95.notnull() & merged_data.basemean_92.notnull()]) / np.mean(merged_data['basemean_92'][merged_data.basemean_95.notnull() & merged_data.basemean_92.notnull()])
        spec95to00 = np.mean(merged_data['basemean_00'][merged_data.basemean_00.notnull() & merged_data.basemean_95.notnull()]) / np.mean(merged_data['basemean_95'][merged_data.basemean_00.notnull() & merged_data.basemean_95.notnull()])
        spec00to06 = np.mean(merged_data['basemean_06'][merged_data.basemean_06.notnull() & merged_data.basemean_00.notnull()]) / np.mean(merged_data['basemean_00'][merged_data.basemean_06.notnull() & merged_data.basemean_00.notnull()])
        #print spec92to95, spec95to00, spec00to06

        # fill in NaN
        merged_data.basemean_95.fillna(value=spec92to95 * merged_data.basemean_92, inplace=True)
        merged_data.basemean_00.fillna(value=spec95to00 * merged_data.basemean_95, inplace=True)
        merged_data.basemean_06.fillna(value=spec00to06 * merged_data.basemean_00, inplace=True)

        area_perf = merged_data[['id', 'transistors', 'basemean_06']][merged_data.transistors.notnull() & merged_data.basemean_06 > 0]

        # return two lists: transistors, specint06 performance
        return area_perf[['transistors', 'basemean_06']].as_matrix()
