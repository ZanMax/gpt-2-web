import Vue from 'vue'
import BootstrapVue from 'bootstrap-vue';
import App from './App.vue'

import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap-vue/dist/bootstrap-vue.css';
import axios from 'axios'
import titleMixin from './mixins/titleMixin'

const baseURL = 'https://textgen.co';
axios.defaults.baseURL = baseURL;

Vue.config.productionTip = false

Vue.use(BootstrapVue);
Vue.mixin(titleMixin)

new Vue({
  render: h => h(App),
}).$mount('#app')
