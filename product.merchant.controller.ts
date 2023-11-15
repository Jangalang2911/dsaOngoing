import express from 'express'
import { ProductMerchantService } from '../services/product.merchant.service'
import { ProductMerchant } from '../database/models/product.merchant'
import { ProductMerchantValidator } from '../validators/product.merchant.validator'
import { uuid } from '../domain.types/miscellaneous/system.types'
import { ApiError } from '../common/api.error'
import { ResponseHandler } from '../common/response.handler'


export class ProductMerchantController{

    private _service   : ProductMerchantService 
    private _validator : ProductMerchantValidator


    constructor(){
        this._service = new ProductMerchantService();
        this._validator = new ProductMerchantValidator();
    }


    public create = async(req: express.Request, res: express.Response) => {
        try{
            const createModel = await ProductMerchantValidator.validateCreateRequest(req.body)
            const pm = await this._service.create(createModel)

            if(pm == null){
                throw new ApiError(500, "Could not create ProductMerchant!")
            }

            ResponseHandler.success(req, res, "ProductMerchant created successfully", 201,
            {ProductMerchant: pm})

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }


    public login = async(req: express.Request, res: express.Response) => {
        try{
            const token = await this._service.login(req)
            if(token == null){
                throw new ApiError(500, "Invalid username or password")
            }
            ResponseHandler.success(req, res, "Logged in successfully", 201, 
            {Token : token})

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }



    public get = async(req: express.Request, res: express.Response) => {
        try {
            const pms = await this._service.get()

            ResponseHandler.success(req, res, "ProductMerchants fetched successfully", 201,
            {ProductMerchants : pms})

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }



    public getById = async(req: express.Request, res: express.Response) => {
        try {
            const pm = await this._service.getById(req.params.id)

            if(pm == null){
                throw new ApiError(500, "Could not find ProductMerchant!")
            }

            ResponseHandler.success(req, res, "ProductMerchant fetched successfully", 201,
            {ProductMerchant : pm})

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }


    public update = async(req: express.Request, res: express.Response) => {
        try {
            const updateModel = await ProductMerchantValidator.validateUpdateRequest(req.body)
            const pm = await this._service.getById(req.params.id)
            if(pm == null){
                throw new ApiError(500, "Could not find ProductMerchant!")
            }
            const updated = await this._service.update(req.params.id, updateModel)
            if(updated == null){
                throw new ApiError(500, "Could not update ProductMerchant!")
            }
            ResponseHandler.success(req, res, "ProductMerchant updated successfully", 201,
            {ProductMerchant : updated})

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }


    public delete = async(req: express.Request, res: express.Response) => {
        try {
            const pm = await this._service.getById(req.params.id)
            if(pm == null){
                throw new ApiError(500, "ProductMerchant does not exist!")
            }

            const deleted = await this._service.delete(req.params.id)
            if(deleted == null){
                throw new ApiError(500, "ProductMerchant could not be deleted!")
            }
            ResponseHandler.success(req, res, "ProductMerchant deleted successfully", 201, null)

        }catch(error){
            ResponseHandler.HandleError(req, res, error)
        }
    }


}