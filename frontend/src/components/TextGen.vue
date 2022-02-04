<template>
    <div>
        <Header/>
        <Sidebar/>
        <div>
            <loading :active.sync="isLoading"
                     :can-cancel="false"
                     :is-full-page="fullPage">
            </loading>

            <b-container class="textgen-container">
                <b-row>
                    <b-col cols="8">
                        <b-form-input v-model="text" placeholder="Enter text"></b-form-input>
                    </b-col>
                    <b-col>
                        <b-form-select v-model="selected" :options="options"></b-form-select>
                    </b-col>
                    <b-col>
                        <b-button variant="outline-primary" @click="GenerateText">Generate</b-button>
                    </b-col>
                </b-row>

                <b-row class="result_textarea">
                    <b-col>
                        <b-form-textarea
                                id="textarea"
                                v-model="result"
                                rows="6"
                                max-rows="20"
                                disabled
                        ></b-form-textarea>
                    </b-col>
                </b-row>
            </b-container>

        </div>
        <Footer/>
    </div>
</template>

<script>
    import Header from "@/components/Header";
    import Sidebar from "@/components/Sidebar";
    import Footer from "@/components/Footer";
    import axios from 'axios';
    import Loading from 'vue-loading-overlay';
    import 'vue-loading-overlay/dist/vue-loading.css';

    export default {
        title: 'Text Generator',
        name: "TextGen",
        data() {
            return {
                text: '',
                result: '',
                selected: 100,
                isLoading: false,
                fullPage: true,
                options: [
                    {value: 100, text: 100},
                    {value: 250, text: 250},
                    {value: 500, text: 500},
                    {value: 750, text: 750},
                ]
            }
        },
        methods: {
            GenerateText(bvModalEvt) {
                bvModalEvt.preventDefault()
                let payload = {'text': this.text, 'length': this.selected}
                if (this.text) {
                    this.isLoading = true
                    axios.post('/api/generate', payload)
                        .then((res) => {
                            this.result = res.data.generated
                            this.isLoading = false
                        })
                        .catch((error) => {
                            // eslint-disable-next-line
                            console.log(error);
                        });
                }
            }
        },
        components: {
            Header,
            Sidebar,
            Footer,
            Loading
        }
    }
</script>

<style scoped>
    .result_textarea {
        padding-top: 10px;
    }
</style>