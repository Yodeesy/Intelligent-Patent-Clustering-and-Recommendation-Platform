import { createStore } from 'vuex'
import axios from 'axios'

export interface Patent {
  pubNo: string;
  title: string;
  summary: string;
  claims: string;
  pubTime: string;
  countryName: string;
  srcDatabase: string;
}

export interface State {
  patents: Patent[];
  selectedPatents: string[];
  clusterResults: any;
  loading: boolean;
}

export default createStore<State>({
  state: {
    patents: [],
    selectedPatents: [],
    clusterResults: null,
    loading: false
  },
  
  mutations: {
    setPatents(state, patents: Patent[]) {
      state.patents = patents;
    },
    setSelectedPatents(state, patents: string[]) {
      state.selectedPatents = patents;
    },
    setClusterResults(state, results: any) {
      state.clusterResults = results;
    },
    setLoading(state, loading: boolean) {
      state.loading = loading;
    }
  },
  
  actions: {
    async fetchRandomPatents({ commit }, size = 100) {
      commit('setLoading', true);
      try {
        const response = await axios.get(`/api/patents/random?size=${size}`);
        commit('setPatents', response.data);
      } finally {
        commit('setLoading', false);
      }
    },
    
    async clusterPatents({ commit, state }) {
      if (state.selectedPatents.length === 0) return;
      
      commit('setLoading', true);
      try {
        const response = await axios.post('/api/patents/cluster', state.selectedPatents);
        commit('setClusterResults', response.data);
      } finally {
        commit('setLoading', false);
      }
    }
  }
}) 